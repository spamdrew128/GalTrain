use crate::bindings::hip_bindings::{
    hipDeviceSynchronize, hipError_t, hipFree, hipMalloc, hipMemcpy, hipMemcpyKind,
};
use std::stringify;

macro_rules! call {
    ($func_call:expr) => {
        unsafe {
            assert_eq!(hipError_t::hipSuccess, $func_call, stringify!($func_call));
        }
    };
}

pub fn hip_sync() {
    call!(hipDeviceSynchronize());
}

pub fn hip_malloc<T>(num_items: usize) -> *mut T {
    let num_bytes = num_items * std::mem::size_of::<T>();
    let mut mem = std::ptr::null_mut::<T>();
    let mem_ptr = (&mut mem) as *mut *mut T;

    call!(hipMalloc(mem_ptr.cast(), num_bytes));
    hip_sync();

    mem
}

pub fn hip_free<T>(ptr: *mut T) {
    call!(hipFree(ptr.cast()));
}

fn hip_memcpy<T>(dest: *mut T, src: *const T, num_items: usize, kind: hipMemcpyKind) {
    let num_bytes = num_items * std::mem::size_of::<T>();
    call!(hipMemcpy(dest.cast(), src.cast(), num_bytes, kind));
    hip_sync();
}

pub fn hip_memcpy_host_to_device<T>(dest: *mut T, src: *const T, num_items: usize) {
    hip_memcpy(dest, src, num_items, hipMemcpyKind::hipMemcpyHostToDevice);
}

pub fn hip_memcpy_device_to_host_<T>(dest: *mut T, src: *const T, num_items: usize) {
    hip_memcpy(dest, src, num_items, hipMemcpyKind::hipMemcpyDeviceToHost);
}
