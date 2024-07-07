fn main() {
    hip::build();
}

mod hip {
    use bindgen::EnumVariation;
    use std::{env, path::PathBuf};

    const WRAPPER_PATH: &str = "./src/wrapper.h";
    const HIP_PATH: &str = "/opt/rocm";
    const BINDINGS_PATH: &str = "./src/hip_bindings";
    const KERNELS_PATH: &str = "./src/kernals";
    const KERNALS_ASM: &str = "libkernels.a";

    fn hip_lib_bindgen() {
        println!("cargo::rustc-link-lib=dylib=hipblas");
        println!("cargo::rustc-link-lib=dylib=rocblas");
        println!("cargo::rustc-link-lib=dylib=amdhip64");

        println!("cargo::rustc-link-search=native={HIP_PATH}/lib");

        let bindings = bindgen::Builder::default()
            .clang_arg(format!("-I{HIP_PATH}/include"))
            .header(WRAPPER_PATH)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .size_t_is_usize(true)
            .default_enum_style(EnumVariation::Rust {
                non_exhaustive: true,
            })
            .must_use_type("hipError")
            .layout_tests(false)
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(PathBuf::from(BINDINGS_PATH).join("hip_bindings.rs"))
            .expect("Couldn't write bindings!");
    }

    fn kernal_bindgen() {
        let file_names = ["vector_add"];
        let files: Vec<String> = file_names
            .iter()
            .map(|name| format!("{KERNELS_PATH}/{name}.cpp"))
            .collect();

        cc::Build::new()
            .compiler("hipcc")
            .debug(false)
            .opt_level(3)
            .files(files)
            .flag(&format!("--offload-arch=gfx1010"))
            .flag("-munsafe-fp-atomics") // Required since AMDGPU doesn't emit hardware atomics by default
            .compile(KERNALS_ASM);

        let out_dir = PathBuf::from(env::var_os("OUT_DIR").unwrap());
        let kernal_asm = out_dir.join(KERNALS_ASM);

        println!("cargo::rustc-link-search=native={}", kernal_asm.display());

        // let bindings = bindgen::Builder::default()
        //     .clang_arg(format!("-I{}", kernal_asm.display()))
        //     .header(WRAPPER_PATH)
        //     .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        //     .size_t_is_usize(true)
        //     .default_enum_style(EnumVariation::Rust {
        //         non_exhaustive: true,
        //     })
        //     .must_use_type("hipError")
        //     .layout_tests(false)
        //     .generate()
        //     .expect("Unable to generate bindings")
        //     .write_to_file(out_path.join("bindings.rs"))
        //     .expect("Couldn't write bindings!");
    }

    pub fn build() {
        hip_lib_bindgen();
        kernal_bindgen();
    }
}
