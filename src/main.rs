use test::vector_add;

mod bindings;
mod hip;
mod test;

fn main() {
    let mut a: Vec<f32> = vec![];
    let mut b: Vec<f32> = vec![];

    for _ in 0..10 {
        a.push(1.0);
        b.push(3.0);
    }

    unsafe {
        let res = vector_add(a, b);
        println!("{:?}\n", res);
    }
}
