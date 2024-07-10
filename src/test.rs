use crate::bindings::kernal_bindings::AddVecs;

pub fn vector_add(a: Vec<f32>, b: Vec<f32>) -> Vec<f32> {
    assert_eq!(a.len(), b.len());
    let len = a.len();

    let mut dest = vec![0_f32; len];
    unsafe {
        AddVecs(dest.as_mut_ptr(), a.as_ptr(), b.as_ptr(), len);
    }
    dest
}
