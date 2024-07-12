#![allow(warnings)]
#![allow(approx_constant)]

pub(crate) mod hip {
    include!(concat!(env!("OUT_DIR"), "/hip_bindings.rs"));
}

pub(crate) mod kernals {
    include!(concat!(env!("OUT_DIR"), "/kernal_bindings.rs"));
}
