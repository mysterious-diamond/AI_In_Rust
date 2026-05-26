use crate::{lmhead::LMHead, tokenizer::Embedding, transformer::TransformerBlock};

pub struct Model {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    lm_head: LMHead,
}

impl Model {
    pub fn new() -> Model {
        let mut blocks: Vec<TransformerBlock> = Vec::new();
        let embedding: Embedding = Embedding::new();
        let lm_head: LMHead = LMHead::new();
        for _ in 0..4 {
            blocks.push(TransformerBlock::new());
        }
        Model {
            embedding,
            blocks,
            lm_head,
        }
    }

    pub fn forward(&self, tokens: &[u8]) -> Vec<Vec<f32>> {
        let embed_tokens: Vec<Vec<f32>> = self.embedding.forward(&tokens);
        let mut result: Vec<Vec<f32>> = embed_tokens.to_vec();
        for block in &self.blocks {
            result = block.forward(&result);
        }
        let result: Vec<Vec<f32>> = self.lm_head.forward(&result);
        result
    }

    fn get_attention_layer_weights(&self, weights: &mut Vec<f32>, transform: &TransformerBlock) {
        for row in &transform.attention_layer.query {
            for value in row {
                weights.push(*value);
            }
        }

        for row in &transform.attention_layer.key {
            for value in row {
                weights.push(*value);
            }
        }

        for row in &transform.attention_layer.value {
            for value in row {
                weights.push(*value);
            }
        }
    }

    fn get_first_layernorm_weights(&self, weights: &mut Vec<f32>, transform: &TransformerBlock) {
        for val in &transform.first_layer_norm.gamma {
            weights.push(*val);
        }

        for val in &transform.first_layer_norm.beta {
            weights.push(*val);
        }
    }

    fn get_feedforward_weights(&self, weights: &mut Vec<f32>, transform: &TransformerBlock) {
        for row in &transform.feed_forward.expansion_weights {
            for val in row {
                weights.push(*val);
            }
        }

        for row in &transform.feed_forward.contraction_weights {
            for val in row {
                weights.push(*val);
            }
        }
    }

    fn get_second_layernorm_weights(&self, weights: &mut Vec<f32>, transform: &TransformerBlock) {
        for val in &transform.second_layer_norm.gamma {
            weights.push(*val);
        }

        for val in &transform.second_layer_norm.beta {
            weights.push(*val);
        }
    }

    fn get_lm_head_weights(&self, weights: &mut Vec<f32>) {
        for row in &self.lm_head.weights {
            for val in row {
                weights.push(*val);
            }
        }
    }

    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights: Vec<f32> = Vec::new();

        for row in &self.embedding.table {
            for value in row {
                weights.push(*value);
            }
        }

        for transform in &self.blocks {
            self.get_attention_layer_weights(&mut weights, transform);
            self.get_first_layernorm_weights(&mut weights, transform);
            self.get_feedforward_weights(&mut weights, transform);
            self.get_second_layernorm_weights(&mut weights, transform);
            self.get_lm_head_weights(&mut weights);
        }

        weights
    }

    pub fn set_weights(&self, weights: &Vec<f32>) {
        let mut id: u64 = 0;
    }
}
