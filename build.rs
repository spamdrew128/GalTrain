/*
    THIS FILE WAS WRITTEN USING https://github.com/jw1912/bullet HEAVILY AS A REFERENCE.
    THANK YOU JW :)
*/

use std::{env, path::PathBuf};

use bindgen::{
    callbacks::{MacroParsingBehavior, ParseCallbacks},
    EnumVariation,
};

const HIP_WRAPPER_PATH: &str = "./src/hip_wrapper.h";
const HIP_PATH: &str = "/opt/rocm";
const HIP_BINDINGS_NAME: &str = "hip_bindings.rs";

const KERNAL_WRAPPER_PATH: &str = "./src/kernals/kernal_wrapper.h";
const KERNELS_PATH: &str = "./src/kernals";
const KERNAL_ASM: &str = "libkernels.a";
const KERNAL_BINDINGS_NAME: &str = "kernal_bindings.rs";

fn out_dir() -> PathBuf {
    PathBuf::from(env::var_os("OUT_DIR").unwrap())
}

// NOTE: THIS IS TO AVOID DUPLICATING DEFINITIONS
const IGNORED_MACROS: &[&str] = &[
    "FP_INFINITE",
    "FP_NAN",
    "FP_NORMAL",
    "FP_SUBNORMAL",
    "FP_ZERO",
    "IPPORT_RESERVED",
];

#[derive(Debug)]
struct CustomParseCallBacks;

impl ParseCallbacks for CustomParseCallBacks {
    fn will_parse_macro(&self, name: &str) -> MacroParsingBehavior {
        if IGNORED_MACROS.contains(&name) {
            MacroParsingBehavior::Ignore
        } else {
            MacroParsingBehavior::Default
        }
    }

    // redirect to normal handler
    fn include_file(&self, filename: &str) {
        bindgen::CargoCallbacks::new().include_file(filename)
    }
}

fn hip_lib_bindgen() {
    println!("cargo::rustc-link-lib=dylib=amdhip64");

    println!("cargo::rustc-link-search=native={HIP_PATH}/lib");

    bindgen::Builder::default()
        .clang_arg(format!("-I{HIP_PATH}/include"))
        .header(HIP_WRAPPER_PATH)
        .parse_callbacks(Box::new(CustomParseCallBacks))
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: true,
        })
        .must_use_type("hipError")
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_dir().join(HIP_BINDINGS_NAME))
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
        .flag("--offload-arch=gfx1010")
        .flag("-munsafe-fp-atomics") // Required since AMDGPU doesn't emit hardware atomics by default
        .compile(KERNAL_ASM);

    println!("cargo::rustc-link-lib=dylib=amdhip64");

    println!("cargo::rustc-link-search=native={HIP_PATH}/lib");

    bindgen::Builder::default()
        .header(KERNAL_WRAPPER_PATH)
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        .size_t_is_usize(true)
        .default_enum_style(EnumVariation::Rust {
            non_exhaustive: true,
        })
        .must_use_type("hipError")
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_dir().join(KERNAL_BINDINGS_NAME))
        .expect("Couldn't write bindings!");
}

pub fn main() {
    hip_lib_bindgen();
    kernal_bindgen();
}
