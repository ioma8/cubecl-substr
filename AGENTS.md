# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs`: Core GPU kernel (`#[cube]`), public API (`find_on_gpu`, helpers) and unit tests.
- `src/main.rs`: Minimal CLI/example that prints the first match.
- `src/tests.rs`: Additional tests targeting the GPU kernel path.
- `benches/find_benchmark.rs`: Criterion benchmark comparing GPU vs `memchr`.
- `Cargo.toml`/`Cargo.lock`: Rust crate configuration (edition 2021, CubeCL + WGPU backend).

## Build, Test, and Development Commands
- Build (debug/release): `cargo build` / `cargo build --release`.
- Run example: `cargo run --release`.
- Tests: `cargo test` (runs unit and module tests under `src/`).
- Benchmarks: `cargo bench` (runs Criterion in `benches/`).
- Lint/format: `cargo fmt --all` and `cargo clippy --all-targets --all-features -D warnings`.

## Coding Style & Naming Conventions
- Follow idiomatic Rust with `rustfmt` defaults; run formatting before commits.
- Enable Clippy and fix warnings; treat warnings as errors in CI/PRs.
- Naming: functions `snake_case`, types `CamelCase`, constants `SCREAMING_SNAKE_CASE`.
- GPU kernels use `#[cube]` and should keep arguments typed as `Array<Line<...>>` with clear, prefix-free names.

## Testing Guidelines
- Framework: Rust built-in test harness; place quick unit tests next to code (`#[cfg(test)]`).
- GPU-dependent tests are in `src/tests.rs`; keep inputs small and deterministic.
- Naming: prefix with `test_` and assert first-match semantics (e.g., `Some(10)`).
- Run locally: `cargo test`; for performance-sensitive changes, attach `cargo bench` output.

## Commit & Pull Request Guidelines
- Commits: Conventional Commits (e.g., `feat: add GPU substring search`).
- PRs: clear description, rationale, and scope; link issues; note GPU/WGPU behavior changes.
- Required checks before opening PR: `cargo fmt --all`, `cargo clippy --all-targets --all-features`, `cargo test`, and (if relevant) `cargo bench` summary.
- Include small code snippets or logs in PRs for tricky GPU/kernel changes.

## Security & Environment
- GPU backend: WGPU via `cubecl-wgpu`; ensure compatible GPU/drivers. CPU-only fallbacks are not provided here.
- Prefer `--release` for realistic performance numbers.
- Avoid unchecked kernel launches in new code paths without bounds checks or early exits.
