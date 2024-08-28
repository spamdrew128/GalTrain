use test::test_sq_matmul;

mod bindings;
mod hip;
mod test;

fn main() {
    let n = 1024;
    println!("Evaluating {n}x{n} matmul:");
    test_sq_matmul(n);
}
