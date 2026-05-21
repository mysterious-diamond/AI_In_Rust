use crate::{
    math::{mean, standard_deviation},
    tokenizer::EMBEDDINGDEPTH,
};

pub struct LayerNorm {
    gamma: Vec<f32>,
    beta: Vec<f32>,
}

impl LayerNorm {
    pub fn new() -> LayerNorm {
        let mut gamma: Vec<f32> = Vec::new();
        let mut beta: Vec<f32> = Vec::new();
        for _ in 0..EMBEDDINGDEPTH {
            gamma.push(1.0);
            beta.push(0.0);
        }

        LayerNorm { gamma, beta }
    }

    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for token in tokens {
            let mean: f32 = mean(token);
            let standard_deviation: f32 = standard_deviation(token, mean);

            let mut token_embedding: Vec<f32> = Vec::new();
            for i in 0..EMBEDDINGDEPTH {
                let normalized: f32 = (token[i] - mean) / (standard_deviation + 0.00001);
                token_embedding.push(normalized * self.gamma[i] + self.beta[i]);
            }
            result.push(token_embedding);
        }

        result
    }
}
