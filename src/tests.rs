#[cfg(test)]
mod tests {
    use cubecl::prelude::*;
    use cubecl_wgpu::{WgpuDevice, WgpuRuntime};
    use crate::mark_matches;

    #[test]
    fn test_mark_matches() {
        let haystack: Vec<u32> = b"The quick brown fox jumps over the lazy dog"
            .iter()
            .map(|&x| x as u32)
            .collect();
        let needle: Vec<u32> = b"brown".iter().map(|&x| x as u32).collect();
        let candidates = haystack.len() - needle.len() + 1;

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

        let mut first: Option<usize> = None;
        for (i, &f) in flags_host.iter().enumerate() {
            if f == 1 {
                first = Some(i);
                break;
            }
        }

        assert_eq!(first, Some(10));
    }
}