// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors
//! Disposable Linux research sidecar. Never a canonical format or production cache.
//! Caller supplies a private root bound to ONE immutable index. No concurrent writes
//! or truncation are permitted during mapped runs. Mapping is read-only/random advice.

use super::*;
use std::{
    fs,
    os::fd::AsRawFd,
    path::{Component, Path, PathBuf},
    ptr::NonNull,
};

#[derive(serde::Serialize, serde::Deserialize)]
struct Directory {
    length: usize,
    offsets: Vec<u32>,
}

struct BufferOwner(Buffer);
impl AsRef<[u8]> for BufferOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_slice()
    }
}
pub(super) fn owned_buffer(buffer: Buffer) -> bytes::Bytes {
    bytes::Bytes::from_owner(BufferOwner(buffer))
}

fn location(source: Option<&str>, token: u32) -> Result<PathBuf> {
    let root = std::env::var_os("FTS_RESEARCH_POSITIONS_ROOT").ok_or_else(|| {
        Error::invalid_input("research sidecar requires a private index-bound root")
    })?;
    let source = source.ok_or_else(|| {
        Error::invalid_input("research sidecar requires modern partition identity")
    })?;
    if !Path::new(source)
        .components()
        .all(|c| matches!(c, Component::Normal(_)))
    {
        return Err(Error::invalid_input("invalid research partition path"));
    }
    Ok(PathBuf::from(root).join(source).join(token.to_string()))
}

pub(super) fn export(
    source: Option<&str>,
    token: u32,
    offsets: &[u32],
    bytes: &[u8],
) -> Result<()> {
    if !std::env::var("FTS_RESEARCH_POSITIONS_MODE").is_ok_and(|m| m == "export") {
        return Ok(());
    }
    let path = location(source, token)?;
    fs::create_dir_all(path.parent().unwrap())?;
    fs::write(path.with_extension("data"), bytes)?;
    fs::write(
        path.with_extension("json"),
        serde_json::to_vec(&Directory {
            length: bytes.len(),
            offsets: offsets.to_vec(),
        })?,
    )?;
    Ok(())
}

struct Mapping {
    pointer: NonNull<u8>,
    len: usize,
}
// SAFETY: private immutable files; no mutable alias is exposed. The mapping outlives
// all Bytes clones through Bytes::from_owner. The experiment forbids file mutation.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}
impl AsRef<[u8]> for Mapping {
    fn as_ref(&self) -> &[u8] {
        // SAFETY: mmap returned len readable bytes, unmapped only in Drop.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }
}
impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: exactly the original mmap region, uniquely owned here.
        unsafe {
            libc::munmap(self.pointer.as_ptr().cast(), self.len);
        }
    }
}

fn load(path: &Path, codec: PositionStreamCodec) -> Result<SharedPositionStream> {
    // Metadata reads and JSON decode are deliberately inside query/cache-miss cost.
    let directory: Directory = serde_json::from_slice(&fs::read(path.with_extension("json"))?)?;
    let file = fs::File::open(path.with_extension("data"))?;
    if file.metadata()?.len() != directory.length as u64
        || directory.offsets.first().is_some_and(|&x| x != 0)
        || directory.offsets.windows(2).any(|w| w[0] > w[1])
        || directory
            .offsets
            .last()
            .is_some_and(|&x| x as usize > directory.length)
        || (directory.length > 0 && directory.offsets.is_empty())
    {
        return Err(Error::index("invalid research position directory"));
    }
    let bytes = if directory.length == 0 {
        bytes::Bytes::new()
    } else {
        // SAFETY: read-only private mapping of checked nonempty immutable file.
        let pointer = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                directory.length,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if pointer == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error().into());
        }
        let pointer = NonNull::new(pointer.cast::<u8>()).expect("non-null mmap region");
        let advice = match std::env::var("FTS_RESEARCH_POSITIONS_MODE").as_deref() {
            Ok("map-normal") => libc::MADV_NORMAL,
            Ok("map-sequential") => libc::MADV_SEQUENTIAL,
            _ => libc::MADV_RANDOM,
        };
        let result = unsafe { libc::madvise(pointer.as_ptr().cast(), directory.length, advice) };
        if result != 0 {
            unsafe {
                libc::munmap(pointer.as_ptr().cast(), directory.length);
            }
            return Err(std::io::Error::last_os_error().into());
        }
        bytes::Bytes::from_owner(Mapping {
            pointer,
            len: directory.length,
        })
    };
    // Keep existing cache accounting conservative: full logical bytes, not a zero
    // weight for mmap. Actual RSS/page-cache occupancy is independently measured.
    trace_term(&bytes, path.to_str().unwrap(), 0);
    Ok(SharedPositionStream::new(codec, directory.offsets, bytes))
}

pub(super) fn read_mapped(
    source: Option<&str>,
    token: u32,
    codec: PositionStreamCodec,
) -> Result<Option<SharedPositionStream>> {
    if !std::env::var("FTS_RESEARCH_POSITIONS_MODE")
        .is_ok_and(|m| matches!(m.as_str(), "map" | "map-normal" | "map-sequential"))
    {
        return Ok(None);
    }
    Ok(Some(load(&location(source, token)?, codec)?))
}

// Traces run separately from performance measurements. No per-block output is
// written in ordinary runs; all modes share the same disabled cached check.
fn trace(value: serde_json::Value) {
    static OUTPUT: std::sync::OnceLock<Option<std::sync::Mutex<fs::File>>> =
        std::sync::OnceLock::new();
    if let Some(output) = OUTPUT.get_or_init(|| {
        std::env::var_os("FTS_RESEARCH_POSITION_TRACE").map(|path| {
            std::sync::Mutex::new(fs::File::create(path).expect("create research trace"))
        })
    }) {
        use std::io::Write;
        let mut file = output.lock().unwrap();
        writeln!(file, "{value}").expect("write research trace");
    }
}
fn tracing_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("FTS_RESEARCH_POSITION_TRACE").is_some())
}
pub(super) fn trace_term(bytes: &bytes::Bytes, source: &str, token: u32) {
    if tracing_enabled() {
        trace(
            serde_json::json!({"event":"term","pointer":bytes.as_ptr() as usize,"source":source,"token":token,"length":bytes.len()}),
        );
    }
}
pub(super) fn trace_block(pointer: usize, start: usize, end: usize) {
    if tracing_enabled() {
        trace(serde_json::json!({"event":"block","pointer":pointer,"start":start,"end":end}));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mapped_storage_preserves_boundaries_and_lifetime() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("term");
        let data: Vec<u8> = (0..9000).map(|x| (x % 251) as u8).collect();
        fs::write(path.with_extension("data"), &data).unwrap();
        fs::write(
            path.with_extension("json"),
            serde_json::to_vec(&Directory {
                length: data.len(),
                offsets: vec![0, 100, 4090, 4200],
            })
            .unwrap(),
        )
        .unwrap();
        // Codec is opaque here; the native reference/codec tests cover decoding.
        let codec = PositionStreamCodec::default();
        let stream = load(&path, codec).unwrap();
        let clone = stream.clone();
        drop(stream);
        assert_eq!(clone.block(1), &data[100..4090]);
        assert_eq!(clone.block(2), &data[4090..4200]);
        assert_eq!(clone.block(3), &data[4200..]);
        // Representative Sophon boundary: the actual Lance Positions codec
        // serializes the entire stream and offers no row-selective decoding.
        use lance_core::cache::{CacheCodec, CacheCodecImpl, CacheEntryReader, CacheEntryWriter};
        let entry = Positions(CompressedPositionStorage::SharedStream(clone));
        let mut body = Vec::new();
        entry
            .serialize(&mut CacheEntryWriter::new(&mut body))
            .unwrap();
        assert!(body.len() >= data.len());
        assert!(!CacheCodec::from_impl::<Positions>().supports_row_selection());
        let body = bytes::Bytes::from(body);
        let restored = Positions::deserialize(&mut CacheEntryReader::new(
            &body,
            0,
            Positions::CURRENT_VERSION,
        ))
        .unwrap();
        let CompressedPositionStorage::SharedStream(restored) = restored.0 else {
            panic!("shared stream expected")
        };
        assert_eq!(restored.bytes(), data);
        fs::write(
            path.with_extension("json"),
            b"{\"length\":9000,\"offsets\":[0,9001]}",
        )
        .unwrap();
        assert!(load(&path, codec).is_err());
        fs::write(
            path.with_extension("json"),
            b"{\"length\":9001,\"offsets\":[0]}",
        )
        .unwrap();
        assert!(load(&path, codec).is_err());
    }
}
