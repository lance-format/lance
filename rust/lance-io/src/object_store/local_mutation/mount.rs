// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::fs::File;
use std::io;
#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
use std::os::fd::AsRawFd;

#[cfg(any(target_os = "linux", target_os = "android"))]
use nix::sys::statfs::{
    BTRFS_SUPER_MAGIC, EXT4_SUPER_MAGIC, F2FS_SUPER_MAGIC, FsType, OVERLAYFS_SUPER_MAGIC,
    TMPFS_MAGIC,
};

// Linux reports the registered driver name in mountinfo and the superblock
// magic in statfs. ext2/3/4 share a magic. nix omits XFS on musl and OHOS.
#[cfg(any(target_os = "linux", target_os = "android"))]
const LOCAL_TYPES: &[(&[u8], FsType)] = &[
    (b"ext2", EXT4_SUPER_MAGIC),
    (b"ext3", EXT4_SUPER_MAGIC),
    (b"ext4", EXT4_SUPER_MAGIC),
    (b"xfs", FsType(0x5846_5342)),
    (b"btrfs", BTRFS_SUPER_MAGIC),
    (b"f2fs", F2FS_SUPER_MAGIC),
    (b"overlay", OVERLAYFS_SUPER_MAGIC),
    (b"tmpfs", TMPFS_MAGIC),
];

#[cfg(any(target_os = "linux", target_os = "android"))]
pub(super) fn supports_inode_lock(file: &File) -> io::Result<bool> {
    // statfs can ask a FUSE daemon to calculate the whole volume's usage.
    // procfs identifies the opened mount without asking it for space statistics.
    let read_mount = || {
        let fdinfo =
            std::fs::read_to_string(format!("/proc/thread-self/fdinfo/{}", file.as_raw_fd()))?;
        let mounts = io::BufReader::new(File::open("/proc/thread-self/mountinfo")?);
        linux_mount_type(&fdinfo, mounts)
    };
    linux_supports_inode_lock(read_mount(), || {
        Ok(nix::sys::statfs::fstatfs(file)?.filesystem_type())
    })
}

#[cfg(any(target_os = "linux", target_os = "android"))]
fn linux_supports_inode_lock(
    mount: io::Result<Vec<u8>>,
    magic: impl FnOnce() -> io::Result<FsType>,
) -> io::Result<bool> {
    match mount {
        // rootfs can be backed by either ramfs or tmpfs.
        Ok(name) if name == b"rootfs" => return Ok(magic()? == TMPFS_MAGIC),
        Ok(name) => return Ok(LOCAL_TYPES.iter().any(|(local, _)| *local == name)),
        // A chroot may lack procfs, or its root may hide the tag's mountinfo row.
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound
                    | io::ErrorKind::PermissionDenied
                    | io::ErrorKind::Unsupported
            ) => {}
        Err(e) => return Err(e),
    }
    // 9p can report the server's native filesystem magic. Without a driver
    // name, magic alone can only rule out native locking.
    let magic = magic()?;
    if LOCAL_TYPES.iter().any(|(_, local)| *local == magic) {
        return Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "cannot identify the tag filesystem for inode locking: expose procfs and the dataset mount inside the process root (e.g. bind-mount the dataset), or coordinate tag writers externally and set local_tag_external_coordination=true",
        ));
    }
    Ok(false)
}

#[cfg(any(test, target_os = "linux", target_os = "android"))]
fn linux_mount_type(fdinfo: &str, mut mounts: impl io::BufRead) -> io::Result<Vec<u8>> {
    let id = fdinfo
        .lines()
        .find_map(|line| line.strip_prefix("mnt_id:"))
        .ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "kernel fdinfo has no mount ID"))?
        .trim_ascii()
        .as_bytes();
    let mut line = Vec::new();
    while mounts.read_until(b'\n', &mut line)? != 0 {
        let mut fields = line.split(|c| *c == b' ');
        if fields.next() == Some(id) {
            return fields
                .find(|field| *field == b"-")
                .and_then(|_| fields.next())
                .filter(|name| !name.is_empty())
                .map(<[u8]>::to_vec)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "missing filesystem type in mountinfo",
                    )
                });
        }
        line.clear();
    }
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "tag mount is absent from /proc/thread-self/mountinfo",
    ))
}

#[cfg(target_vendor = "apple")]
pub(super) fn supports_inode_lock(file: &File) -> io::Result<bool> {
    // With ATTR_VOL_INFO, XNU returns the vnode tag of the mount root, including
    // for descriptors opened on regular files. No space statistics are requested.
    let mut attrs = nix::libc::attrlist {
        bitmapcount: nix::libc::ATTR_BIT_MAP_COUNT,
        reserved: 0,
        commonattr: nix::libc::ATTR_CMN_OBJTAG,
        volattr: nix::libc::ATTR_VOL_INFO,
        dirattr: 0,
        fileattr: 0,
        forkattr: 0,
    };
    let mut reply = [0_u32; 2]; // byte length, then fsobj_tag_t
    // SAFETY: attrs and reply are live, aligned buffers of the declared sizes.
    let status = unsafe {
        nix::libc::fgetattrlist(
            file.as_raw_fd(),
            (&raw mut attrs).cast(),
            reply.as_mut_ptr().cast(),
            size_of_val(&reply),
            0,
        )
    };
    if status == -1 {
        return Err(io::Error::last_os_error());
    }
    if reply[0] as usize != size_of_val(&reply) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "missing volume vnode tag",
        ));
    }
    Ok(supports_volume_tag(reply[1]))
}

#[cfg(target_vendor = "apple")]
fn supports_volume_tag(tag: u32) -> bool {
    // Public vtagtype values in Darwin's <sys/vnode.h>.
    const VT_HFS: u32 = 16;
    const VT_APFS: u32 = 25;
    matches!(tag, VT_HFS | VT_APFS)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
pub(super) fn supports_inode_lock(_file: &File) -> io::Result<bool> {
    Ok(false)
}

#[cfg(test)]
mod tests {
    use super::linux_mount_type;
    #[cfg(any(target_os = "linux", target_os = "android"))]
    use super::{EXT4_SUPER_MAGIC, FsType, TMPFS_MAGIC, linux_supports_inode_lock};
    #[cfg(any(target_os = "linux", target_os = "android"))]
    use nix::sys::statfs::NFS_SUPER_MAGIC;
    use rstest::rstest;
    use std::io::ErrorKind;

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest]
    #[case::nfs(b"nfs4", false)]
    #[case::smb(b"cifs", false)]
    #[case::remote_fuse(b"fuse.rclone", false)]
    #[case::plan9(b"9p", false)]
    #[case::ext2(b"ext2", true)]
    #[case::ext3(b"ext3", true)]
    #[case::ext4(b"ext4", true)]
    fn test_local_mutation_mount_type(#[case] name: &[u8], #[case] supported: bool) {
        let actual = linux_supports_inode_lock(Ok(name.to_vec()), || {
            panic!("the mount driver name is sufficient")
        });
        assert_eq!(actual.unwrap(), supported);
    }

    #[cfg(target_vendor = "apple")]
    #[rstest]
    #[case::nfs(2, false)]
    #[case::smb(23, false)]
    #[case::hfs(16, true)]
    #[case::apfs(25, true)]
    fn test_local_mutation_volume_tag(#[case] tag: u32, #[case] supported: bool) {
        assert_eq!(super::supports_volume_tag(tag), supported);
    }

    #[rstest]
    #[case::nfs("31", b"nfs4")]
    #[case::fuse("32", b"fuse.rclone")]
    #[case::local("310", b"ext4")]
    fn test_local_mutation_mount_id(#[case] id: &str, #[case] expected: &[u8]) {
        let mounts = b"310 1 8:1 / /data rw - ext4 /dev/sda rw\n\
                       31 310 0:1 / /data/\xff rw shared:7 - nfs4 host:/data rw\n\
                       32 310 0:2 / /data/mirror rw - fuse.rclone remote rw\n";
        let fdinfo = format!("pos:\t0\nmnt_id:\t{id}\nino:\t13\n");
        assert_eq!(linux_mount_type(&fdinfo, &mounts[..]).unwrap(), expected);
    }

    #[test]
    fn test_local_mutation_mount_outside_process_root() {
        assert_eq!(
            linux_mount_type("mnt_id:\t71\n", &b""[..])
                .unwrap_err()
                .kind(),
            ErrorKind::Unsupported
        );
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest]
    #[case::missing_procfs(ErrorKind::NotFound)]
    #[case::denied_procfs(ErrorKind::PermissionDenied)]
    #[case::hidden_mount(ErrorKind::Unsupported)]
    fn test_local_mutation_unknown_driver(#[case] kind: ErrorKind) {
        // ext4 and 9p exporting an ext4 directory can return the same magic.
        let err = linux_supports_inode_lock(Err(kind.into()), || Ok(EXT4_SUPER_MAGIC)).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Unsupported);
        assert!(
            err.to_string()
                .contains("local_tag_external_coordination=true")
        );
        assert!(!linux_supports_inode_lock(Err(kind.into()), || Ok(NFS_SUPER_MAGIC)).unwrap());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[rstest]
    #[case::tmpfs(TMPFS_MAGIC, true)]
    #[case::ramfs(FsType(0x8584_58f6_u32 as _), false)]
    fn test_local_mutation_rootfs(#[case] magic: FsType, #[case] supported: bool) {
        let result = linux_supports_inode_lock(Ok(b"rootfs".to_vec()), || Ok(magic));
        assert_eq!(result.unwrap(), supported);
    }
}
