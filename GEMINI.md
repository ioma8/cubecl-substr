# Gemini Code Assistant Context

## Project Overview

This project is a Rust application that demonstrates how to use the CubeCL library to perform a substring search on a byte array using the GPU. The core logic involves a GPU kernel that marks all potential starting positions of a `needle` within a `haystack`, and a CPU-side scan to find the first marked position. This approach simplifies the GPU kernel by avoiding a more complex reduction operation on the GPU itself.

The main technologies used are:
*   **Rust:** The programming language used for the project.
*   **CubeCL:** A crate for writing and executing GPU kernels.
*   **cubecl-wgpu:** The WGPU backend for CubeCL.

## Building and Running

To build and run the project, use the following command:

```bash
cargo run --release
```

## Development Conventions

*   The project follows standard Rust conventions.
*   The main logic is contained within `src/main.rs`.
*   The GPU kernel is defined using the `#[cube]` macro from CubeCL.
*   The application initializes a WGPU device, allocates memory on the GPU for the haystack, needle, and a flags array, launches the kernel, reads the flags back to the CPU, and then scans the flags to find the first match.
*   After every change run `cargo check` to ensure it builds correctly
*   In folder `cubec-examples` there are official examples from cubecl crate which you need to check if you meet any issues with our code
