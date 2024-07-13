use test::test_sq_matmul;

mod bindings;
mod hip;
mod test;

fn main() {
    for n in 0..1223 {
        println!("{n}");
        test_sq_matmul(n);
    }
}
