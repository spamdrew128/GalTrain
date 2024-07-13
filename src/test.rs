use crate::{
    bindings::kernals::{AddVecs, SqMatMul},
    hip::hip_funcs::{
        hip_free, hip_malloc, hip_memcpy_device_to_host, hip_memcpy_host_to_device, hip_sync,
    },
};

pub unsafe fn vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
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

    dest
}

fn matmul_cpu_verify(gpu_res: &[f32], a: &[f32], b: &[f32], n: usize) {
    for r in 0..n {
        for c in 0..n {
            let mut dot_product = 0.0;
            for i in 0..n {
                dot_product += a[r * n + i] * b[i * n + c];
            }
            assert_eq!(dot_product, gpu_res[r * n + c]);
        }
    }
}

pub fn test_sq_matmul(n: usize) {
    let len = n * n;
    let mut a: Vec<f32> = vec![];
    let mut b: Vec<f32> = vec![];
    let mut dest = vec![0_f32; len];

    for i in 0..len {
        a.push(0.6 * i as f32);
        b.push(1.2 * i as f32);
    }

    let d_a: *mut f32 = hip_malloc(len);
    let d_b: *mut f32 = hip_malloc(len);
    let d_dest: *mut f32 = hip_malloc(len);

    hip_memcpy_host_to_device(d_a, a.as_ptr(), len);
    hip_memcpy_host_to_device(d_b, b.as_ptr(), len);

    unsafe {
        SqMatMul(d_dest, d_a, d_b, n);

        hip_memcpy_device_to_host(dest.as_mut_ptr(), d_dest, len);

        hip_free(d_a);
        hip_free(d_b);
        hip_free(d_dest);
    }

    matmul_cpu_verify(dest.as_slice(), a.as_slice(), b.as_slice(), n);
}
