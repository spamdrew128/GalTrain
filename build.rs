fn main() {
    hip::build();
}

mod hip {
    use std::env;
    use std::path::PathBuf;
    use bindgen::EnumVariation;

    const WRAPPER_PATH: &str = "./src/wrapper.h";
    const HIP_PATH: &str = "/opt/rocm";

    pub fn build() {
        println!("cargo::rustc-link-lib=dylib=hipblas");
        println!("cargo::rustc-link-lib=dylib=rocblas");
        println!("cargo::rustc-link-lib=dylib=amdhip64");

        println!("cargo::rustc-link-search=native={HIP_PATH}/lib");

        // The bindgen::Builder is the main entry point
        // to bindgen, and lets you build up options for
        // the resulting bindings.
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
            .expect("Unable to generate bindings");

        // Write the bindings to the $OUT_DIR/bindings.rs file.
        let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
        bindings
            .write_to_file(out_path.join("bindings.rs"))
            .expect("Couldn't write bindings!");
    }
}
