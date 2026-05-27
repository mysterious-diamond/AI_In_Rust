use crate::{model::Model, tokenizer::Tokenizer};

pub fn generate(
    prompt: &str,
    model: &Model,
    tokenizer: &Tokenizer,
    amount_of_tokens: u32,
) -> Vec<u8> {
    let mut input: Vec<u8> = tokenizer.encode(prompt);

    for _ in 0..amount_of_tokens {
        println!("generating token...");
        let result: Vec<Vec<f32>> = model.forward(&input);
        let mut max: f32 = 0.0;
        let mut id: u8 = 0;
        for i in 0..256 {
            if result[result.len() - 1][i] > max {
                id = i as u8;
                max = result[result.len() - 1][i];
            }
        }

        input.push(id);
    }

    input
}
