use crate::{
    attention::Attention, feedforward::FeedForward, layernorm::LayerNorm, math::add_elementwise,
};

pub struct TransformerBlock {
    attention_layer: Attention,
    first_layer_norm: LayerNorm,
    feed_forward: FeedForward,
    second_layer_norm: LayerNorm,
}

impl TransformerBlock {
    pub fn new() -> TransformerBlock {
        let attention_layer: Attention = Attention::new();
        let first_layer_norm: LayerNorm = LayerNorm::new();
        let feed_forward: FeedForward = FeedForward::new();
        let second_layer_norm: LayerNorm = LayerNorm::new();
        TransformerBlock {
            attention_layer,
            first_layer_norm,
            feed_forward,
            second_layer_norm,
        }
    }

    pub fn forward(&self, tokens: &[Vec<f32>]) -> Vec<Vec<f32>> {
        let result: Vec<Vec<f32>> = self.first_layer_norm.forward(&add_elementwise(
            &self.attention_layer.forward(tokens),
            tokens,
        ));

        let result: Vec<Vec<f32>> = self.second_layer_norm.forward(&add_elementwise(
            &self.feed_forward.forward(&result),
            &result,
        ));

        result
    }
}
