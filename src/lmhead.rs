use rand::{RngExt, rng};

use crate::{
    math::{matmul, softmax},
    tokenizer::EMBEDDINGDEPTH,
};

pub struct LMHead {
    weights: Vec<Vec<f32>>,
}

impl LMHead {
    pub fn new() -> LMHead {
        let mut weights: Vec<Vec<f32>> = Vec::new();
        for _ in 0..256 {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..EMBEDDINGDEPTH {
                let bound: f32 = 1.0 / (EMBEDDINGDEPTH as f32).sqrt();
                let random_val: f32 = rng().random_range(-bound..bound);
                row.push(random_val);
            }
            weights.push(row);
        }
        LMHead { weights }
    }

    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            result.push(softmax(matmul(token, &self.weights)));
        }

        result
    }
}
