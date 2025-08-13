use criterion::{criterion_group, criterion_main, Criterion};
use cubecl_substr::find_on_gpu_with_client;
use memchr::memmem;
use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

fn criterion_benchmark(c: &mut Criterion) {
    let haystack: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
    let needle: Vec<u8> = (0..1000).map(|i| (i % 256) as u8).collect();

    let haystack_u32: Vec<u32> = haystack.iter().map(|&x| x as u32).collect();
    let needle_u32: Vec<u32> = needle.iter().map(|&x| x as u32).collect();

    let mut group = c.benchmark_group("find_string");

    group.bench_function("find_on_gpu", |b| {
        let client = WgpuRuntime::client(&WgpuDevice::default());
        b.iter(|| {
            find_on_gpu_with_client(&client, std::hint::black_box(&haystack_u32), std::hint::black_box(&needle_u32));
        })
    });

    group.bench_function("memchr", |b| {
        b.iter(|| {
            memmem::find(std::hint::black_box(&haystack), std::hint::black_box(&needle));
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
