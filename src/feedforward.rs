use rand::{RngExt, rng};

use crate::{
    math::{matmul, relu},
    tokenizer::EMBEDDINGDEPTH,
};

pub struct FeedForward {
    pub expansion_weights: Vec<Vec<f32>>,
    pub contraction_weights: Vec<Vec<f32>>,
}

impl FeedForward {
    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();

        for row in tokens {
            let expanded_row: Vec<f32> = relu(matmul(row, &self.expansion_weights));
            let result_row: Vec<f32> = matmul(&expanded_row, &self.contraction_weights);
            result.push(result_row);
        }

        result
    }

    pub fn new() -> FeedForward {
        let mut expansion_weights: Vec<Vec<f32>> = Vec::new();
        let mut contraction_weights: Vec<Vec<f32>> = Vec::new();

        let scale: f32 = 1.0 / (EMBEDDINGDEPTH as f32).sqrt();
        for _ in 0..EMBEDDINGDEPTH {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..4 * EMBEDDINGDEPTH {
                row.push(rng().random_range(-scale..scale));
            }
            expansion_weights.push(row);
        }

        let scale: f32 = 1.0 / (4.0 * EMBEDDINGDEPTH as f32).sqrt();
        for _ in 0..4 * EMBEDDINGDEPTH {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..EMBEDDINGDEPTH {
                row.push(rng().random_range(-scale..scale));
            }
            contraction_weights.push(row);
        }

        FeedForward {
            expansion_weights,
            contraction_weights,
        }
    }
}
