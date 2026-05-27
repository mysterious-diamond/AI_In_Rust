use crate::{lmhead::LMHead, tokenizer::Embedding, transformer::TransformerBlock};

pub struct Model {
    embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    lm_head: LMHead,
}

fn get_embedding_weights(weights: &mut Vec<f32>, embedding: &Embedding) {
    for row in &embedding.table {
        for value in row {
            weights.push(*value);
        }
    }
}

fn set_embedding_weights(weights: &Vec<f32>, embedding: &mut Embedding, id: &mut usize) {
    for row in &mut embedding.table {
        for value in row {
            *value = weights[*id];
            *id += 1;
        }
    }
}

fn get_attention_layer_weights(weights: &mut Vec<f32>, transform: &TransformerBlock) {
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

fn set_attention_layer_weights(
    weights: &Vec<f32>,
    transform: &mut TransformerBlock,
    id: &mut usize,
) {
    for row in &mut transform.attention_layer.query {
        for value in row {
            *value = weights[*id];
            *id += 1;
        }
    }

    for row in &mut transform.attention_layer.key {
        for value in row {
            *value = weights[*id];
            *id += 1;
        }
    }

    for row in &mut transform.attention_layer.value {
        for value in row {
            *value = weights[*id];
            *id += 1;
        }
    }
}

fn get_first_layernorm_weights(weights: &mut Vec<f32>, transform: &TransformerBlock) {
    for val in &transform.first_layer_norm.gamma {
        weights.push(*val);
    }

    for val in &transform.first_layer_norm.beta {
        weights.push(*val);
    }
}

fn set_first_layernorm_weights(
    weights: &Vec<f32>,
    transform: &mut TransformerBlock,
    id: &mut usize,
) {
    for val in &mut transform.first_layer_norm.gamma {
        *val = weights[*id];
        *id += 1;
    }

    for val in &mut transform.first_layer_norm.beta {
        *val = weights[*id];
        *id += 1;
    }
}

fn get_feedforward_weights(weights: &mut Vec<f32>, transform: &TransformerBlock) {
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

fn set_feedforward_weights(weights: &Vec<f32>, transform: &mut TransformerBlock, id: &mut usize) {
    for row in &mut transform.feed_forward.expansion_weights {
        for val in row {
            *val = weights[*id];
            *id += 1;
        }
    }

    for row in &mut transform.feed_forward.contraction_weights {
        for val in row {
            *val = weights[*id];
            *id += 1;
        }
    }
}

fn get_second_layernorm_weights(weights: &mut Vec<f32>, transform: &TransformerBlock) {
    for val in &transform.second_layer_norm.gamma {
        weights.push(*val);
    }

    for val in &transform.second_layer_norm.beta {
        weights.push(*val);
    }
}

fn set_second_layernorm_weights(
    weights: &Vec<f32>,
    transform: &mut TransformerBlock,
    id: &mut usize,
) {
    for val in &mut transform.second_layer_norm.gamma {
        *val = weights[*id];
        *id += 1;
    }

    for val in &mut transform.second_layer_norm.beta {
        *val = weights[*id];
        *id += 1;
    }
}

fn get_lm_head_weights(weights: &mut Vec<f32>, lm_head: &LMHead) {
    for row in &lm_head.weights {
        for val in row {
            weights.push(*val);
        }
    }
}

fn set_lm_head_weights(weights: &Vec<f32>, lm_head: &mut LMHead, id: &mut usize) {
    for row in &mut lm_head.weights {
        for val in row {
            *val = weights[*id];
            *id += 1;
        }
    }
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

    pub fn get_weights(&self) -> Vec<f32> {
        let mut weights: Vec<f32> = Vec::new();

        get_embedding_weights(&mut weights, &self.embedding);

        for transform in &self.blocks {
            get_attention_layer_weights(&mut weights, transform);
            get_first_layernorm_weights(&mut weights, transform);
            get_feedforward_weights(&mut weights, transform);
            get_second_layernorm_weights(&mut weights, transform);
        }
        get_lm_head_weights(&mut weights, &self.lm_head);

        weights
    }

    pub fn set_weights(&mut self, weights: &Vec<f32>) {
        let mut id: usize = 0;

        set_embedding_weights(&weights, &mut self.embedding, &mut id);

        for transform in &mut self.blocks {
            set_attention_layer_weights(&weights, transform, &mut id);
            set_first_layernorm_weights(&weights, transform, &mut id);
            set_feedforward_weights(&weights, transform, &mut id);
            set_second_layernorm_weights(&weights, transform, &mut id);
        }
        set_lm_head_weights(&weights, &mut self.lm_head, &mut id);
    }
}
