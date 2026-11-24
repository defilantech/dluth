//! Build-time work for the CUDA crate:
//! 1. compile the kernels to a cubin (GPU machine code for one architecture) with nvcc;
//! 2. generate Rust declarations for the CUDA driver API from NVIDIA's header.
//!
//! Both need the CUDA toolkit, which is why this crate is not a default workspace member.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

/// The GPU architecture the kernels are compiled for: compute capability (major, minor).
/// The library reads the same pair back as `BUILT_FOR`, so the two cannot drift apart.
const BUILT_FOR: (u32, u32) = (12, 1);

fn main() {
    let cuda = env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
    println!("cargo:rerun-if-changed=kernels/vecadd.cu");
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");

    let arch = format!("sm_{}{}", BUILT_FOR.0, BUILT_FOR.1);
    std::fs::write(
        out.join("built_for.rs"),
        format!(
            "/// The compute capability (major, minor) the embedded kernels were compiled for.\n\
             pub const BUILT_FOR: (u32, u32) = ({}, {});\n",
            BUILT_FOR.0, BUILT_FOR.1
        ),
    )
    .expect("write built_for.rs");

    let nvcc = format!("{cuda}/bin/nvcc");
    let status = Command::new(&nvcc)
        .args(["-cubin", "-O3", &format!("-arch={arch}"), "-o"])
        .arg(out.join("kernels.cubin"))
        .arg("kernels/vecadd.cu")
        .status()
        .unwrap_or_else(|e| panic!("could not run {nvcc}: {e}. Set CUDA_HOME to the CUDA toolkit directory."));
    assert!(status.success(), "nvcc failed to compile kernels/vecadd.cu for {arch}");

    let mut builder = bindgen::Builder::default().header("wrapper.h").clang_arg(format!("-I{cuda}/include"));
    // A machine with only the libclang runtime lacks clang's own freestanding headers
    // (stddef.h and friends). The C compiler's copies serve the same purpose.
    if let Ok(o) = Command::new("cc").arg("-print-file-name=include").output() {
        let dir = String::from_utf8_lossy(&o.stdout).trim().to_string();
        if Path::new(&dir).join("stddef.h").exists() {
            builder = builder.clang_arg(format!("-I{dir}"));
        }
    }
    builder
        .allowlist_function(
            "cu(Init|DeviceGet|DeviceGetCount|DeviceGetAttribute|DevicePrimaryCtxRetain|DevicePrimaryCtxRelease\
             |CtxSetCurrent|CtxSynchronize|ModuleLoadData|ModuleUnload|ModuleGetFunction|MemAlloc|MemFree\
             |MemcpyHtoD|MemcpyDtoH|LaunchKernel|GetErrorString)(_v2)?",
        )
        .allowlist_type("CU.*")
        .allowlist_var("CU_.*")
        // Enum values keep the names the vendor documentation uses (CUDA_SUCCESS, ...).
        .prepend_enum_name(false)
        .layout_tests(false)
        .generate_comments(false)
        .generate()
        .expect("bindgen could not generate CUDA driver bindings from cuda.h")
        .write_to_file(out.join("sys.rs"))
        .expect("write sys.rs");

    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=dylib=cuda");
}
