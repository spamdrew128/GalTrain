use crate::hip::{
        blas::BlasHandle,
        standard::{hip_calloc, hip_free, hip_malloc, hip_memcpy_device_to_host},
    };

fn matmul_cpu_verify(gpu_res: &[f32], a: &[f32], b: &[f32], n: usize) {
    for r in 0..n {
        for c in 0..n {
            let mut dot_product = 0.0;
            for i in 0..n {
                dot_product += a[r * n + i] * b[i * n + c];
            }
            assert!(
                (dot_product - gpu_res[r * n + c]).abs() < 0.001,
                "Fail at row {r} col {c}\n"
            );
        }
    }
}

pub fn test_matmul(n: usize) {
    let len = n * n;
    let mut a: Vec<f32> = vec![];
    let mut b: Vec<f32> = vec![];
    let mut dest = vec![0_f32; len];

    
    for i in 0..len {
        let v = i as f32;
        a.push(1.0 + v);
        b.push(v);
    }
    
    let stopwatch = std::time::Instant::now();
    let d_a: *mut f32 = hip_malloc(len);
    let d_b: *mut f32 = hip_malloc(len);
    let d_dest: *mut f32 = hip_calloc(len);
    
    let handle = BlasHandle::new();
    handle.basic_matmul(n, n, n, d_a, d_b, d_dest);
    hip_memcpy_device_to_host(dest.as_mut_ptr(), d_dest, len);

    hip_free(d_a);
    hip_free(d_b);
    hip_free(d_dest);

    println!("GPU took {} ms", stopwatch.elapsed().as_millis());

    let stopwatch = std::time::Instant::now();
    matmul_cpu_verify(dest.as_slice(), a.as_slice(), b.as_slice(), n);
    println!("CPU took {} ms", stopwatch.elapsed().as_millis());
}
