use crate::{model::Model, tokenizer::Tokenizer};

mod attention;
mod feedforward;
mod inference;
mod layernorm;
mod lmhead;
mod loss;
mod math;
mod model;
mod tokenizer;
mod train;
mod transformer;

fn main() {
    let tokenizer: Tokenizer = Tokenizer::new();
    let mut model: Model = Model::new();

    let tokens: Vec<u8> = std::fs::read("./prompt.txt").expect("File not found");
    let window_size = 16;
    for i in 0..10 {
        let input = &tokens[i..i + window_size];
        let target = &tokens[i + 1..i + window_size + 1];

        println!("Step {}", i);
        train::train(&mut model, input, target, 1e-4, 0.01);
    }

    let generated = inference::generate("hel", &model, &tokenizer, 5);
    println!("{}", tokenizer.decode(&generated));
}
