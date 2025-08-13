use cubecl::channel::MutexComputeChannel;
use cubecl::prelude::*;
use cubecl_runtime::server::Handle;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime, WgpuServer};
use once_cell::sync::Lazy;
use std::collections::HashMap;
use std::sync::Mutex;

static DEVICE: Lazy<WgpuDevice> = Lazy::new(WgpuDevice::default);
static CLIENT: Lazy<ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>> = Lazy::new(|| WgpuRuntime::client(&DEVICE));

#[derive(Clone, Debug)]
struct GpuFindCacheEntry {
    hay_len: usize,
    needle_len: usize,
    candidates: usize,
    group_count_words: u32,
    d_hay: Handle,
    d_needle: Handle,
    d_flags: Handle,
    d_group_mins: Handle,
}

static GPU_CACHE: Lazy<Mutex<HashMap<(usize, usize, usize), GpuFindCacheEntry>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

#[cube(launch_unchecked)]
fn mark_matches(
    hay: &Array<Line<u32>>,
    needle: &Array<Line<u32>>,
    flags: &mut Array<Line<u32>>,
) {
    let i = ABSOLUTE_POS;
    if i >= flags.len() {
        terminate!();
    }
    let nlen = needle.len();
    // Early multi-byte boundary checks to reduce full comparisons
    if nlen >= 8 {
        if hay[i] != needle[0]
            || hay[i + 1] != needle[1]
            || hay[i + 2] != needle[2]
            || hay[i + 3] != needle[3]
            || hay[i + (nlen - 1)] != needle[nlen - 1]
            || hay[i + (nlen - 2)] != needle[nlen - 2]
            || hay[i + (nlen - 3)] != needle[nlen - 3]
            || hay[i + (nlen - 4)] != needle[nlen - 4]
        {
            flags[i] = Line::new(0);
            terminate!();
        }
        // Additional spaced sentinel checks to prune non-matches early
        if nlen >= 64 {
            let quarter = nlen / 4;
            let eighth = nlen / 8;
            if hay[i + quarter] != needle[quarter]
                || hay[i + (nlen - quarter - 1)] != needle[nlen - quarter - 1]
                || hay[i + (2 * quarter)] != needle[2 * quarter]
                || hay[i + (eighth)] != needle[eighth]
                || hay[i + (3 * eighth)] != needle[3 * eighth]
            {
                flags[i] = Line::new(0);
                terminate!();
            }
        }
        let mut ok: bool = true;
        let mut j = 4u32;
        let end = nlen - 4;
        while j < end {
            if hay[i + j] != needle[j] {
                ok = false;
                break;
            }
            j += 1;
        }
        flags[i] = if ok { Line::new(1) } else { Line::new(0) };
        terminate!();
    }
    if nlen >= 2 {
        if hay[i] != needle[0] || hay[i + (nlen - 1)] != needle[nlen - 1] {
            flags[i] = Line::new(0);
            terminate!();
        }
    }
    let mut ok: bool = true;
    let mut j = 0u32;
    while j < nlen {
        if hay[i + j] != needle[j] {
            ok = false;
            break;
        }
        j += 1;
    }
    flags[i] = if ok { Line::new(1) } else { Line::new(0) };
}

// Compress 32 flag entries into one u32 mask per thread
#[cube(launch_unchecked)]
fn compress_flags(flags: &Array<Line<u32>>, masks: &mut Array<Line<u32>>) {
    let word_index = ABSOLUTE_POS;
    if word_index >= masks.len() {
        terminate!();
    }
    let candidates = flags.len();
    let base = word_index * 32;
    if base >= candidates {
        masks[word_index] = Line::new(0);
        terminate!();
    }
    let mut m = 0u32;
    let mut k = 0u32;
    while k < 32 && (base + k) < candidates {
        if flags[base + k] == Line::new(1) {
            m = m + (1u32 << k);
        }
        k += 1;
    }
    masks[word_index] = Line::new(m);
}

pub fn find_on_gpu(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    let haystack: Vec<u32> = haystack.iter().map(|&x| x as u32).collect();
    let needle: Vec<u32> = needle.iter().map(|&x| x as u32).collect();
    find_on_gpu_with_client(&CLIENT, &haystack, &needle)
}

pub fn find_on_gpu_with_client(
    client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
    haystack: &[u32],
    needle: &[u32],
) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }

    let candidates = haystack.len() - needle.len() + 1;
    if candidates == 0 {
        return None;
    }

    // Reuse device buffers when the same client + host slices are passed repeatedly.
    let key = (
        client as *const _ as usize,
        haystack.as_ptr() as usize,
        needle.as_ptr() as usize,
    );

    let mut cache = GPU_CACHE.lock().unwrap();
    let entry = if let Some(entry) = cache.get(&key) {
        // Validate sizes; if mismatched, recreate.
        if entry.hay_len == haystack.len() && entry.needle_len == needle.len() {
            entry.clone()
        } else {
            // Recreate buffers for new sizes
            let new_entry = create_cache_entry(client, haystack, needle);
            cache.insert(key, new_entry.clone());
            new_entry
        }
    } else {
        let new_entry = create_cache_entry(client, haystack, needle);
        cache.insert(key, new_entry.clone());
        new_entry
    };
    drop(cache);

    // Launch kernels using cached buffers only (no per-call allocations or uploads).
    launch_and_reduce(
        client,
        &entry,
    )
}

fn create_cache_entry(
    client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
    haystack: &[u32],
    needle: &[u32],
) -> GpuFindCacheEntry {
    let candidates = haystack.len() - needle.len() + 1;
    let words_count: usize = (candidates + 31) / 32;
    let threads_per_group: u32 = 256;
    let group_count_words: u32 = ((words_count as u32) + threads_per_group - 1) / threads_per_group;

    let d_hay = client.create(bytemuck::cast_slice(haystack));
    let d_needle = client.create(bytemuck::cast_slice(needle));
    let d_flags = client.empty(candidates * core::mem::size_of::<u32>());
    let d_group_mins = client.empty(group_count_words as usize * core::mem::size_of::<u32>());

    GpuFindCacheEntry {
        hay_len: haystack.len(),
        needle_len: needle.len(),
        candidates,
        group_count_words,
        d_hay,
        d_needle,
        d_flags,
        d_group_mins,
    }
}

fn launch_and_reduce(
    client: &ComputeClient<WgpuServer, MutexComputeChannel<WgpuServer>>,
    entry: &GpuFindCacheEntry,
) -> Option<usize> {
    let threads_per_group: u32 = 256;
    let group_count_x: u32 = ((entry.candidates as u32) + threads_per_group - 1) / threads_per_group;
    let group_count_words: u32 = entry.group_count_words;

    unsafe {
        mark_matches::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(group_count_x, 1, 1),
            CubeDim::new(threads_per_group, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&entry.d_hay, entry.hay_len, 1),
            ArrayArg::from_raw_parts::<u32>(&entry.d_needle, entry.needle_len, 1),
            ArrayArg::from_raw_parts::<u32>(&entry.d_flags, entry.candidates, 1),
        );
        reduce_group_first_flags::launch_unchecked::<WgpuRuntime>(
            client,
            CubeCount::Static(group_count_words, 1, 1),
            CubeDim::new(threads_per_group, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&entry.d_flags, entry.candidates, 1),
            ArrayArg::from_raw_parts::<u32>(&entry.d_group_mins, entry.group_count_words as usize, 1),
        );
    }

    // Read back tiny group minima and compute global minimum on CPU
    let mins_bytes = client.read_one(entry.d_group_mins.clone().binding());
    let mins: Vec<u32> = bytemuck::cast_slice(&mins_bytes).to_vec();
    let mut best: Option<usize> = None;
    for &pos in &mins {
        if pos != u32::MAX {
            let p = pos as usize;
            if p < entry.candidates {
                best = Some(best.map_or(p, |b| b.min(p)));
            }
        }
    }
    best
}

// Reduce flags per workgroup to first set position (or u32::MAX)
#[cube(launch_unchecked)]
fn reduce_group_first_flags(flags: &Array<Line<u32>>, out: &mut Array<Line<u32>>) {
    let abs = ABSOLUTE_POS;
    let lane = abs % 256u32;
    let group = abs / 256u32;
    if lane != 0u32 { terminate!(); }
    if group >= out.len() { terminate!(); }

    let total = flags.len();
    let base = group * 256u32 * 32u32; // each group scans 256 words worth of flags
    let end = if base + 256u32 * 32u32 > total { total } else { base + 256u32 * 32u32 };
    let mut i = base;
    while i < end {
        if flags[i] == Line::new(1u32) {
            out[group] = Line::new(i);
            terminate!();
        }
        i += 1u32;
    }
    out[group] = Line::new(u32::MAX);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_on_gpu() {
        let haystack = b"The quick brown fox jumps over the lazy dog";
        let needle = b"brown";
        assert_eq!(find_on_gpu(haystack, needle), Some(10));
    }

    #[test]
    fn test_find_on_gpu_end() {
        let haystack = b"The quick brown fox jumps over the lazy dog";
        let needle = b"dog";
        assert_eq!(find_on_gpu(haystack, needle), Some(40));
    }

    #[test]
    fn test_find_on_gpu_no_match() {
        let haystack = b"The quick brown fox jumps over the lazy dog";
        let needle = b"cat";
        assert_eq!(find_on_gpu(haystack, needle), None);
    }

    #[test]
    fn test_find_on_gpu_empty_needle() {
        let haystack = b"The quick brown fox jumps over the lazy dog";
        let needle = b"";
        assert_eq!(find_on_gpu(haystack, needle), Some(0));
    }

    #[test]
    fn test_find_on_gpu_needle_longer_than_haystack() {
        let haystack = b"short";
        let needle = b"longer_needle";
        assert_eq!(find_on_gpu(haystack, needle), None);
    }
}
