use test::test_sq_matmul;

mod bindings;
mod hip;
mod test;

fn main() {
    test_sq_matmul(1200);
}
