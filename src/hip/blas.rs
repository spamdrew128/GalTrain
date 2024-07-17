#![allow(non_snake_case)]

use std::ffi::c_int;

use crate::bindings::hip::{
    hipblasCreate, hipblasHandle_t, hipblasOperation_t, hipblasSgemm, hipblasStatus_t,
};

macro_rules! call {
    ($func_call:expr) => {
        unsafe {
            assert_eq!(
                hipblasStatus_t::HIPBLAS_STATUS_SUCCESS,
                $func_call,
                stringify!($func_call)
            );
        }
    };
}

pub struct BlasHandle(hipblasHandle_t);

impl BlasHandle {
    pub fn new() -> Self {
        let mut handle: hipblasHandle_t = std::ptr::null_mut();

        call!(hipblasCreate((&mut handle) as *mut hipblasHandle_t));

        Self(handle)
    }

    pub fn basic_matmul(
        &self,
        m: usize,
        n: usize,
        k: usize,
        A: *const f32,
        B: *const f32,
        C: *mut f32,
    ) {
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;

        // A (MxK) * B (KxN)
        // column major format
        call!(hipblasSgemm(
            self.0,
            hipblasOperation_t::HIPBLAS_OP_N,
            hipblasOperation_t::HIPBLAS_OP_N,
            m as c_int,
            n as c_int,
            k as c_int,
            &alpha,
            A,
            m as c_int,
            B,
            k as c_int,
            &beta,
            C,
            m as c_int,
        ));
    }
}
