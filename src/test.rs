use crate::{
    bindings::kernal_bindings::AddVecs,
    hip::hip_funcs::{hip_malloc, hip_memcpy_host_to_device},
};

pub fn vector_add(a: Vec<f32>, b: Vec<f32>) {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    let d_a: *mut f32 = hip_malloc(len);
    let d_b: *mut f32 = hip_malloc(len);
    let d_dest: *mut f32 = hip_malloc(len);

    hip_memcpy_host_to_device(d_a, a.as_ptr(), len);
    hip_memcpy_host_to_device(d_b, b.as_ptr(), len);

    unsafe {
        AddVecs(d_dest, d_a, d_b, len);
    }
}

// extern "C" void AddVecs(float *dest, const float *a, const float* b, size_t len) {
//     size_t arrBytes = len * sizeof(float);

//     float *d_a, *d_b, *d_dest;
//     H_A(hipMalloc((void**)&d_a, arrBytes));
//     H_A(hipMalloc((void**)&d_b, arrBytes));
//     H_A(hipMalloc((void**)&d_dest, arrBytes));

//     H_A(hipMemcpy(d_a, a, arrBytes, hipMemcpyHostToDevice));
//     H_A(hipMemcpy(d_b, b, arrBytes, hipMemcpyHostToDevice));

//     uint32_t num_blocks = (len + threadsPerBlock - 1) / threadsPerBlock;
//     KernalVectorAdd<<<num_blocks, threadsPerBlock>>>(d_dest, d_a, d_b, len);

//     H_A(hipDeviceSynchronize());

//     H_A(hipMemcpy(dest, d_dest, arrBytes, hipMemcpyDeviceToHost));

//     H_A(hipFree(d_a));
//     H_A(hipFree(d_b));
//     H_A(hipFree(d_dest));
// }
