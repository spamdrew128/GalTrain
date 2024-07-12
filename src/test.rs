use crate::{
    bindings::kernal_bindings::AddVecs,
    hip::hip_funcs::{
        hip_free, hip_malloc, hip_memcpy_device_to_host, hip_memcpy_host_to_device, hip_sync,
    },
};

pub unsafe fn vector_add(a: Vec<f32>, b: Vec<f32>) {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    let d_a: *mut f32 = hip_malloc(len);
    let d_b: *mut f32 = hip_malloc(len);
    let d_dest: *mut f32 = hip_malloc(len);

    hip_memcpy_host_to_device(d_a, a.as_ptr(), len);
    hip_memcpy_host_to_device(d_b, b.as_ptr(), len);

    AddVecs(d_dest, d_a, d_b, len);
    hip_sync();

    let mut dest = vec![0_f32; len];
    hip_memcpy_device_to_host(dest.as_mut_ptr(), d_dest, len);

    hip_free(d_a);
    hip_free(d_b);
    hip_free(d_dest);
}
