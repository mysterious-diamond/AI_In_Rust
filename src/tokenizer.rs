use rand::{self, RngExt};

pub const EMBEDDINGDEPTH: usize = 16;

pub struct Tokenizer {}
pub struct Embedding {
    pub table: Vec<Vec<f32>>,
}

impl Tokenizer {
    pub fn encode(&self, text: &str) -> Vec<u8> {
        let mut tokens: Vec<u8> = Vec::new();
        for character in text.chars() {
            tokens.push(character as u8);
        }

        tokens
    }

    pub fn _decode(&self, tokens: &[u8]) -> String {
        let mut text: String = String::new();
        for token in tokens {
            text.push(*token as char);
        }

        text
    }
    pub fn new() -> Tokenizer {
        Tokenizer {}
    }
}

impl Embedding {
    pub fn compute_positional_encoding(pos: i32) -> Vec<f32> {
        let mut res: Vec<f32> = Vec::new();
        for depth in 0..EMBEDDINGDEPTH {
            if depth % 2 == 0 {
                res.push(
                    ((pos as f32 / 10000_f32.powf(depth as f32 / EMBEDDINGDEPTH as f32)) as f32)
                        .sin(),
                )
            } else {
                res.push(
                    ((pos as f32 / 10000_f32.powf((depth - 1) as f32 / EMBEDDINGDEPTH as f32))
                        as f32)
                        .cos(),
                );
            }
        }

        res
    }

    pub fn new() -> Embedding {
        let mut table: Vec<Vec<f32>> = Vec::new();
        for _ in 0..256 {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..EMBEDDINGDEPTH {
                row.push(rand::rng().random_range(-1.0..1.0));
            }

            table.push(row);
        }

        Embedding { table }
    }

    pub fn forward(&self, tokens: &[u8]) -> Vec<Vec<f32>> {
        let mut result: Vec<Vec<f32>> = Vec::new();
        for i in 0..tokens.len() {
            let token_embed: Vec<f32> = self.table[tokens[i] as usize].clone();
            let position_embed: Vec<f32> = Self::compute_positional_encoding(i as i32);
            let mut combined: Vec<f32> = Vec::new();

            for i in 0..EMBEDDINGDEPTH {
                combined.push(token_embed[i] + position_embed[i]);
            }
            result.push(combined);
        }
        result
    }
}
