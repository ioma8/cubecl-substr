use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

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

pub fn find_on_gpu(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    if needle.len() > haystack.len() {
        return None;
    }

    let haystack: Vec<u32> = haystack.iter().map(|&x| x as u32).collect();
    let needle: Vec<u32> = needle.iter().map(|&x| x as u32).collect();

    let candidates = haystack.len() - needle.len() + 1;
    if candidates == 0 {
        return None;
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let d_hay = client.create(bytemuck::cast_slice(&haystack));
    let d_needle = client.create(bytemuck::cast_slice(&needle));
    let d_flags = client.empty(candidates * core::mem::size_of::<u32>());

    unsafe {
        mark_matches::launch_unchecked::<WgpuRuntime>(
            &client,
            CubeCount::Static(1, 1, 1),
            CubeDim::new(candidates as u32, 1, 1),
            ArrayArg::from_raw_parts::<u32>(&d_hay, haystack.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&d_needle, needle.len(), 1),
            ArrayArg::from_raw_parts::<u32>(&d_flags, candidates, 1),
        );
    }

    let flags_host_bytes = client.read_one(d_flags.binding());
    let flags_host: Vec<u32> = bytemuck::cast_slice(&flags_host_bytes).to_vec();

    for (i, &f) in flags_host.iter().enumerate() {
        if f == 1 {
            return Some(i);
        }
    }

    None
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
