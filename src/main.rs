use cubecl_substr::find_on_gpu;

fn main() {
    let haystack = b"The quick brown fox jumps over the lazy dog";
    let needle = b"brown";

    match find_on_gpu(haystack, needle) {
        Some(idx) => {
            println!("first match at index {}", idx);
            let h_bytes: Vec<u8> = haystack.iter().map(|&x| x).collect();
            println!(
                "matched slice: {:?}",
                &std::str::from_utf8(&h_bytes[idx..idx + needle.len()]).unwrap()
            );
        }
        None => println!("no match"),
    }
}