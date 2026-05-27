use crate::{model::Model, tokenizer::Tokenizer};

mod attention;
mod feedforward;
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

    let text = "hello";
    let input = tokenizer.encode(text);
    let target = tokenizer.encode("ello!");
    let before = loss::average_entropy_loss(&model.forward(&input), &target);
    println!("Loss before: {}", before);

    for step in 0..10 {
        train::train(&mut model, &input, &target, 1e-4, 0.01);
        let loss = loss::average_entropy_loss(&model.forward(&input), &target);
        println!("Step {}: {}", step, loss);
    }
}
