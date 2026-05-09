use crate::{
    math::{dot, matmul, softmax},
    tokenizer::EMBEDDINGDEPTH,
};
use rand::{self, RngExt, rng};

pub struct Attention {
    query: Vec<Vec<f32>>,
    key: Vec<Vec<f32>>,
    value: Vec<Vec<f32>>,
}

impl Attention {
    pub fn new() -> Attention {
        let mut query: Vec<Vec<f32>> = Vec::new();
        let mut key: Vec<Vec<f32>> = Vec::new();
        let mut value: Vec<Vec<f32>> = Vec::new();
        let range: f32 = 1.0 / (EMBEDDINGDEPTH as f32).sqrt();

        for _ in 0..EMBEDDINGDEPTH {
            let mut qrow: Vec<f32> = Vec::new();
            let mut krow: Vec<f32> = Vec::new();
            let mut vrow: Vec<f32> = Vec::new();
            for _ in 0..EMBEDDINGDEPTH {
                qrow.push(rng().random_range(-range..range));
                krow.push(rng().random_range(-range..range));
                vrow.push(rng().random_range(-range..range));
            }

            query.push(qrow);
            key.push(krow);
            value.push(vrow);
        }

        Attention { query, key, value }
    }

    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let mut query: Vec<Vec<f32>> = Vec::new();
        let mut key: Vec<Vec<f32>> = Vec::new();
        let mut value: Vec<Vec<f32>> = Vec::new();

        for token in tokens {
            query.push(matmul(token, &self.query));
            key.push(matmul(token, &self.key));
            value.push(matmul(token, &self.value));
        }

        let scale: f32 = (EMBEDDINGDEPTH as f32).sqrt();
        let mut weights: Vec<Vec<f32>> = Vec::new();
        for i in 0..query.len() {
            let mut scores: Vec<f32> = Vec::new();
            for j in 0..key.len() {
                scores.push(dot(&query[i], &key[j]) / scale);
            }
            weights.push(softmax(scores));
        }

        let mut resulting_weights: Vec<Vec<f32>> = Vec::new();
        for i in 0..tokens.len() {
            let mut result: Vec<f32> = Vec::new();
            for d in 0..EMBEDDINGDEPTH {
                let mut dresult: f32 = 0.0;
                for j in 0..tokens.len() {
                    dresult += weights[i][j] * value[j][d];
                }
                result.push(dresult);
            }
            resulting_weights.push(result);
        }

        resulting_weights
    }
}
