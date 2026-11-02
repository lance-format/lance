// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Memory accounting of the cgroup the current process runs in.

use std::path::{Path, PathBuf};

/// cgroup v1 reports a group without a limit as the largest page-aligned
/// `i64`, so the value depends on the page size. Anything at or above this
/// floor means no limit for pages of up to 1 MiB.
const V1_UNLIMITED_FLOOR: u64 = i64::MAX as u64 & !0xF_FFFF;

/// The memory of the cgroup whose limit applies to the process, in bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct CgroupMemory {
    /// Memory charged to the group that sets the tightest limit, or to the
    /// process's own group when no group sets one.
    pub(super) current_bytes: Option<u64>,
    /// `None` when no group on the way to the root sets a limit.
    pub(super) limit_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Version {
    V1,
    V2,
}

impl Version {
    fn usage_file(self) -> &'static str {
        match self {
            Self::V1 => "memory.usage_in_bytes",
            Self::V2 => "memory.current",
        }
    }

    fn limit_file(self) -> &'static str {
        match self {
            Self::V1 => "memory.limit_in_bytes",
            Self::V2 => "memory.max",
        }
    }
}

/// The groups whose files report the process's memory.
#[derive(Debug, Clone, PartialEq, Eq)]
struct CgroupFiles {
    version: Version,
    /// The process's own group first, then every ancestor up to the mount. A
    /// limit can sit on any of them, and the tightest one applies.
    groups: Vec<PathBuf>,
}

/// Memory of the process's cgroup, or `None` when it is not in a memory cgroup.
#[cfg(all(feature = "metrics", target_os = "linux"))]
pub(super) fn read_cgroup_memory() -> Option<CgroupMemory> {
    // Not cached: a process can be moved to another group while it runs.
    discover(Path::new("/")).as_ref().map(read)
}

/// Finds the process's memory cgroup under `root`, which is `/` outside tests.
fn discover(root: &Path) -> Option<CgroupFiles> {
    let mountinfo = std::fs::read_to_string(root.join("proc/self/mountinfo")).ok()?;
    let membership = std::fs::read_to_string(root.join("proc/self/cgroup")).ok()?;
    // A unified hierarchy without the memory controller (hybrid v1/v2 hosts)
    // has no memory files; the v1 controller reports instead.
    [Version::V2, Version::V1]
        .into_iter()
        .find_map(|version| discover_version(root, &mountinfo, &membership, version))
}

fn discover_version(
    root: &Path,
    mountinfo: &str,
    membership: &str,
    version: Version,
) -> Option<CgroupFiles> {
    let (mount_root, mount_point) = match version {
        Version::V2 => find_mount(mountinfo, "cgroup2", None)?,
        Version::V1 => find_mount(mountinfo, "cgroup", Some("memory"))?,
    };
    let path = membership.lines().find_map(|line| {
        let mut fields = line.splitn(3, ':');
        let hierarchy_id = fields.next()?;
        let controllers = fields.next()?;
        let path = fields.next()?;
        let matches = match version {
            Version::V2 => hierarchy_id == "0" && controllers.is_empty(),
            Version::V1 => controllers.split(',').any(|name| name == "memory"),
        };
        matches.then_some(path)
    })?;
    let mount = root.join(mount_point.trim_start_matches('/'));
    // The mount shows the hierarchy from `mount_root` down: a container
    // without a cgroup namespace mounts its own group, while the membership
    // line still names the host's path.
    let relative = Path::new(path)
        .strip_prefix(mount_root)
        .unwrap_or(Path::new(path.trim_start_matches('/')));
    let own = [mount.join(relative), mount.clone()]
        .into_iter()
        .find(|group| group.join(version.usage_file()).is_file())?;
    let groups = own
        .ancestors()
        .take_while(|group| group.starts_with(&mount))
        .map(Path::to_path_buf)
        .collect();
    Some(CgroupFiles { version, groups })
}

/// Root and mount point of the first `fstype` mount in `/proc/self/mountinfo`,
/// optionally requiring `option` among its super options.
fn find_mount<'a>(
    mountinfo: &'a str,
    fstype: &str,
    option: Option<&str>,
) -> Option<(&'a str, &'a str)> {
    mountinfo.lines().find_map(|line| {
        let (mount_fields, filesystem_fields) = line.split_once(" - ")?;
        let mut filesystem_fields = filesystem_fields.split_whitespace();
        if filesystem_fields.next()? != fstype {
            return None;
        }
        if let Some(option) = option {
            let super_options = filesystem_fields.nth(1)?;
            if !super_options
                .split(',')
                .any(|candidate| candidate == option)
            {
                return None;
            }
        }
        let mut mount_fields = mount_fields.split_whitespace().skip(3);
        Some((mount_fields.next()?, mount_fields.next()?))
    })
}

fn read(files: &CgroupFiles) -> CgroupMemory {
    let version = files.version;
    // An ancestor's usage includes everything below it, so when several groups
    // share the tightest limit the outermost one runs out first.
    let tightest = files
        .groups
        .iter()
        .rev()
        .filter_map(|group| Some((read_limit(version, group)?, group)))
        .min_by_key(|(limit, _)| *limit);
    let usage_group = tightest
        .map(|(_, group)| group)
        .or_else(|| files.groups.first());
    CgroupMemory {
        current_bytes: usage_group.and_then(|group| read_bytes(&group.join(version.usage_file()))),
        limit_bytes: tightest.map(|(limit, _)| limit),
    }
}

fn read_limit(version: Version, group: &Path) -> Option<u64> {
    let limit = read_bytes(&group.join(version.limit_file()))?;
    (version == Version::V2 || limit < V1_UNLIMITED_FLOOR).then_some(limit)
}

/// A byte count file; v2's `max` (no limit) reads as `None`.
fn read_bytes(path: &Path) -> Option<u64> {
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::tempfile::TempStdDir;
    use rstest::rstest;

    const V2_MOUNTINFO: &str = "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n\
        35 25 0:30 / /sys/fs/cgroup rw,nosuid,nodev,noexec,relatime shared:9 - cgroup2 cgroup2 rw,nsdelegate\n";
    const V2_CONTAINER_MOUNTINFO: &str = "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n\
        35 25 0:30 /system.slice/docker-abc.scope /sys/fs/cgroup ro,nosuid shared:9 - cgroup2 cgroup2 rw\n";
    const V1_MOUNTINFO: &str = "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n\
        36 25 0:31 / /sys/fs/cgroup/cpu rw,nosuid shared:10 - cgroup cgroup rw,cpu,cpuacct\n\
        37 25 0:32 / /sys/fs/cgroup/memory rw,nosuid shared:11 - cgroup cgroup rw,memory\n";
    const V1_CONTAINER_MOUNTINFO: &str = "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n\
        37 25 0:32 /docker/abc /sys/fs/cgroup/memory ro,nosuid shared:11 - cgroup cgroup rw,memory\n";
    const HYBRID_MOUNTINFO: &str = "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n\
        35 25 0:30 / /sys/fs/cgroup/unified rw,nosuid shared:9 - cgroup2 cgroup2 rw,nsdelegate\n\
        37 25 0:32 / /sys/fs/cgroup/memory rw,nosuid shared:11 - cgroup cgroup rw,memory\n";
    /// What cgroup v1 reports for a group without a limit with 4 KiB pages.
    const V1_UNLIMITED_4K: &str = "9223372036854771712";
    /// The same with 64 KiB pages.
    const V1_UNLIMITED_64K: &str = "9223372036854710272";

    /// A fake filesystem root holding `files`, as `(path below the root, contents)`.
    fn fake_root(files: &[(&str, &str)]) -> TempStdDir {
        let root = TempStdDir::default();
        for (path, contents) in files {
            let path = root.join(path);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, contents).unwrap();
        }
        root
    }

    #[rstest]
    #[case::v2_own_limit(&[
        ("proc/self/mountinfo", V2_MOUNTINFO),
        ("proc/self/cgroup", "0::/app.slice/lance.scope\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.current", "100\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.max", "1000\n"),
        ("sys/fs/cgroup/app.slice/memory.current", "300\n"),
        ("sys/fs/cgroup/app.slice/memory.max", "max\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(1000) }))]
    #[case::v2_tighter_ancestor(&[
        ("proc/self/mountinfo", V2_MOUNTINFO),
        ("proc/self/cgroup", "0::/app.slice/lance.scope\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.current", "100\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.max", "1000\n"),
        ("sys/fs/cgroup/app.slice/memory.current", "300\n"),
        ("sys/fs/cgroup/app.slice/memory.max", "500\n"),
    ], Some(CgroupMemory { current_bytes: Some(300), limit_bytes: Some(500) }))]
    #[case::v2_shared_limit_reports_the_outer_group(&[
        ("proc/self/mountinfo", V2_MOUNTINFO),
        ("proc/self/cgroup", "0::/app.slice/lance.scope\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.current", "100\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.max", "500\n"),
        ("sys/fs/cgroup/app.slice/memory.current", "300\n"),
        ("sys/fs/cgroup/app.slice/memory.max", "500\n"),
    ], Some(CgroupMemory { current_bytes: Some(300), limit_bytes: Some(500) }))]
    #[case::v2_unlimited(&[
        ("proc/self/mountinfo", V2_MOUNTINFO),
        ("proc/self/cgroup", "0::/app.slice/lance.scope\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.current", "100\n"),
        ("sys/fs/cgroup/app.slice/lance.scope/memory.max", "max\n"),
        ("sys/fs/cgroup/app.slice/memory.current", "300\n"),
        ("sys/fs/cgroup/app.slice/memory.max", "max\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: None }))]
    #[case::v2_container_host_path(&[
        ("proc/self/mountinfo", V2_MOUNTINFO),
        ("proc/self/cgroup", "0::/system.slice/docker-abc.scope\n"),
        ("sys/fs/cgroup/memory.current", "100\n"),
        ("sys/fs/cgroup/memory.max", "800\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(800) }))]
    #[case::v2_nested_in_container(&[
        ("proc/self/mountinfo", V2_CONTAINER_MOUNTINFO),
        ("proc/self/cgroup", "0::/system.slice/docker-abc.scope/worker\n"),
        ("sys/fs/cgroup/worker/memory.current", "100\n"),
        ("sys/fs/cgroup/worker/memory.max", "500\n"),
        ("sys/fs/cgroup/memory.current", "300\n"),
        ("sys/fs/cgroup/memory.max", "800\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(500) }))]
    #[case::v1_limited(&[
        ("proc/self/mountinfo", V1_MOUNTINFO),
        ("proc/self/cgroup", "5:cpu,cpuacct:/lance\n4:memory:/lance\n0::/lance\n"),
        ("sys/fs/cgroup/memory/lance/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/lance/memory.limit_in_bytes", "1000\n"),
        ("sys/fs/cgroup/memory/memory.usage_in_bytes", "300\n"),
        ("sys/fs/cgroup/memory/memory.limit_in_bytes", V1_UNLIMITED_4K),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(1000) }))]
    #[case::v1_tighter_ancestor(&[
        ("proc/self/mountinfo", V1_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/app/lance\n"),
        ("sys/fs/cgroup/memory/app/lance/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/app/lance/memory.limit_in_bytes", V1_UNLIMITED_4K),
        ("sys/fs/cgroup/memory/app/memory.usage_in_bytes", "300\n"),
        ("sys/fs/cgroup/memory/app/memory.limit_in_bytes", "500\n"),
    ], Some(CgroupMemory { current_bytes: Some(300), limit_bytes: Some(500) }))]
    #[case::v1_unlimited(&[
        ("proc/self/mountinfo", V1_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/lance\n"),
        ("sys/fs/cgroup/memory/lance/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/lance/memory.limit_in_bytes", V1_UNLIMITED_4K),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: None }))]
    #[case::v1_unlimited_with_64k_pages(&[
        ("proc/self/mountinfo", V1_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/lance\n"),
        ("sys/fs/cgroup/memory/lance/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/lance/memory.limit_in_bytes", V1_UNLIMITED_64K),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: None }))]
    #[case::v1_container_host_path(&[
        ("proc/self/mountinfo", V1_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/docker/abc\n"),
        ("sys/fs/cgroup/memory/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/memory.limit_in_bytes", "800\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(800) }))]
    #[case::v1_nested_in_container(&[
        ("proc/self/mountinfo", V1_CONTAINER_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/docker/abc/worker\n"),
        ("sys/fs/cgroup/memory/worker/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/worker/memory.limit_in_bytes", "500\n"),
        ("sys/fs/cgroup/memory/memory.usage_in_bytes", "300\n"),
        ("sys/fs/cgroup/memory/memory.limit_in_bytes", "800\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(500) }))]
    #[case::hybrid_reports_through_v1(&[
        ("proc/self/mountinfo", HYBRID_MOUNTINFO),
        ("proc/self/cgroup", "4:memory:/lance\n0::/lance\n"),
        ("sys/fs/cgroup/unified/lance/cgroup.procs", "1\n"),
        ("sys/fs/cgroup/memory/lance/memory.usage_in_bytes", "100\n"),
        ("sys/fs/cgroup/memory/lance/memory.limit_in_bytes", "1000\n"),
    ], Some(CgroupMemory { current_bytes: Some(100), limit_bytes: Some(1000) }))]
    #[case::no_cgroup_mount(&[
        ("proc/self/mountinfo", "25 1 259:9 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p9 rw\n"),
        ("proc/self/cgroup", "0::/\n"),
    ], None)]
    fn test_discover_and_read(
        #[case] files: &[(&str, &str)],
        #[case] expected: Option<CgroupMemory>,
    ) {
        let root = fake_root(files);
        assert_eq!(discover(&root).as_ref().map(read), expected);
    }
}
