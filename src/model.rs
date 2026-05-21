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
}
