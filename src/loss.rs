pub fn cross_entropy_loss(probabilities: &[f32], expected: u8) -> f32 {
    -((probabilities[expected as usize] + 0.0000000001).ln())
}

pub fn average_entropy_loss(probabilities: &[Vec<f32>], expecteds: &[u8]) -> f32 {
    let mut sum: f32 = 0.0;
    let mut amount: f32 = 0.0;
    for i in 0..probabilities.len() {
        sum += cross_entropy_loss(&probabilities[i], expecteds[i]);
        amount += 1.0;
    }

    sum / amount
}
