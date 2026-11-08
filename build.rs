#[cfg(feature = "hip")]
fn main() {
    use std::{env, path::PathBuf};
    assert_eq!(
        env::var("CARGO_CFG_TARGET_OS").unwrap(),
        "linux",
        "HIP requires Linux"
    );
    for key in ["HIP_PLATFORM", "ROCM_PATH", "CUDA_PATH", "GPU_ARCH"] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let rocm = PathBuf::from(env::var_os("ROCM_PATH").unwrap_or_else(|| "/opt/rocm".into()));
    let platform = env::var("HIP_PLATFORM").unwrap_or_else(|_| "amd".into());
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .compiler(rocm.join("bin/hipcc"))
        .file("src/tdigest/gpu/kernels.hip");
    match platform.as_str() {
        "amd" => {
            build.flag("-x").flag("hip").flag("-ffp-contract=off");
            if let Ok(arch) = env::var("GPU_ARCH") {
                build.flag(format!("--offload-arch={arch}"));
            }
            println!("cargo:rustc-link-search=native={}/lib", rocm.display());
        }
        "nvidia" => {
            let cuda = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".into());
            build
                .include(PathBuf::from(&cuda).join("include"))
                .include(rocm.join("include"))
                .define("__HIP_PLATFORM_NVIDIA__", None)
                .flag("--forward-unknown-to-host-compiler")
                .flag("-Xcompiler=-fPIC")
                .flag("-x")
                .flag("cu")
                .flag("--fmad=false")
                .flag("--ftz=false");
            if let Ok(arch) = env::var("GPU_ARCH") {
                build.flag(format!("-arch={arch}"));
            }
            println!("cargo:rustc-link-search=native={cuda}/lib64");
        }
        _ => panic!("HIP_PLATFORM must be amd or nvidia"),
    }
    println!("cargo:rerun-if-changed=src/tdigest/gpu/kernels.hip");
    println!("cargo:rerun-if-changed=src/tdigest/gpu/training.hpp");
    println!("cargo:rerun-if-changed=src/tdigest/gpu/merge.hpp");
    println!("cargo:rerun-if-changed=src/tdigest/gpu/compress.hpp");
    build.compile("gr_tdigest_hip");
    println!(
        "cargo:rustc-link-lib={}",
        if platform == "amd" {
            "amdhip64"
        } else {
            "cudart"
        }
    );
}

#[cfg(not(feature = "hip"))]
fn main() {}
