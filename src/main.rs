use crate::{model::Model, tokenizer::Tokenizer};

mod attention;
mod feedforward;
mod layernorm;
mod lmhead;
mod math;
mod model;
mod tokenizer;
mod transformer;

fn main() {
    let tokenizer: Tokenizer = Tokenizer::new();
    let model: Model = Model::new();
    let encoded: Vec<u8> = tokenizer.encode("hello");
    let output: Vec<Vec<f32>> = model.forward(&encoded);
    let sum: f32 = output[0].iter().sum();

    println!("{}", sum);
    println!("{:?}", output[0].len());
}
