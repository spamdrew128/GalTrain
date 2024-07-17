use test::test_matmul;

mod bindings;
mod hip;
mod test;

fn main() {
    let n = 3400;
    println!("Evaluating {n}x{n} matmul:");
    test_matmul(n);
}
