use crate::tokenizer::EMBEDDINGDEPTH;

pub fn softmax(tokens: Vec<f32>) -> Vec<f32> {
    let mut res: Vec<f32> = Vec::new();
    let mut sum: f32 = 0.0;
    for token in tokens {
        let result = f32::exp(token);
        res.push(result);
        sum += result;
    }

    for i in 0..res.len() {
        res[i] /= sum;
    }
    res
}

pub fn matmul(vec: &[f32], matrix: &[Vec<f32>]) -> Vec<f32> {
    let mut res: Vec<f32> = Vec::new();

    for row in matrix {
        let mut result: f32 = 0.0;
        for i in 0..EMBEDDINGDEPTH {
            result += vec[i] * row[i];
        }

        res.push(result);
    }
    res
}

pub fn dot(first: &[f32], second: &[f32]) -> f32 {
    let mut result: f32 = 0.0;
    for i in 0..EMBEDDINGDEPTH {
        result += first[i] * second[i];
    }
    result
}

pub fn relu(vec: Vec<f32>) -> Vec<f32> {
    let mut relu_result: Vec<f32> = Vec::new();
    for element in vec {
        relu_result.push(if element > 0.0 { element } else { 0.0 });
    }
    relu_result
}

pub fn mean(vec: &[f32]) -> f32 {
    let mut amount: f32 = 0.0;
    let mut sum: f32 = 0.0;

    for val in vec {
        sum += val;
        amount += 1.0;
    }

    sum / amount
}

pub fn standard_deviation(vec: &[f32], mean: f32) -> f32 {
    let mut amount: f32 = 0.0;
    let mut sum: f32 = 0.0;

    for val in vec {
        sum += (val - mean).powi(2);
        amount += 1.0;
    }

    (sum / amount).sqrt()
}
