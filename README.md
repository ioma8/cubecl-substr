# cubecl-substr

GPU-accelerated substring search in Rust using CubeCL with the WGPU backend. It launches a simple kernel that marks all candidate start positions and then reduces on CPU to find the first match.

## Requirements
- Rust (stable)
- A supported GPU/driver for WGPU (Metal/Vulkan/DX12). CPU fallback is not provided.

## Build & Run
- Build (debug): `cargo build`
- Build (release): `cargo build --release`
- Run example: `cargo run --release`

## Tests & Benchmarks
- Unit tests: `cargo test`
- Criterion benches: `cargo bench` (compares GPU vs memchr)

## Layout
- `src/lib.rs` – GPU kernel (`#[cube]`), public API (`find_on_gpu`) and helpers
- `src/main.rs` – small example that prints the first match
- `src/tests.rs` – GPU kernel test
- `benches/find_benchmark.rs` – benchmark group `find_string`

## Notes
- By design, large buffers benefit more; prefer `--release` for realistic numbers.
- See `AGENTS.md` and `GEMINI.md` for extra guidance and conventions.
