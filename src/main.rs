use crate::attention::Attention;
use crate::feedforward::FeedForward;
use crate::tokenizer::{Embedding, Tokenizer};

mod attention;
mod feedforward;
mod math;
mod tokenizer;

fn main() {
    let tokenizer: Tokenizer = Tokenizer::new();
    let embedding: Embedding = Embedding::new();
    let attention: Attention = Attention::new();
    let fforward: FeedForward = FeedForward::new();

    let encoded: Vec<u8> = tokenizer.encode("hello");
    println!("{:?}", encoded);

    let embed: Vec<Vec<f32>> = embedding.forward(&encoded);
    println!("{:?}", embed);

    let attent: Vec<Vec<f32>> = attention.forward(&embed);
    println!("{:?}", attent);

    let ff: Vec<Vec<f32>> = fforward.forward(&attent);
    println!("{:?}", ff);
}
