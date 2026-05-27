use crate::{loss::average_entropy_loss, model::Model};

pub fn train(model: &mut Model, input: &[u8], target: &[u8], epsilon: f32, learning_rate: f32) {
    let mut weights = model.get_weights();

    for i in 0..weights.len() {
        weights[i] += epsilon;
        model.set_weights(&weights);
        let predictions_up = model.forward(input);
        let loss_up = average_entropy_loss(&predictions_up, target);

        weights[i] -= 2.0 * epsilon;
        model.set_weights(&weights);
        let predictions_down = model.forward(input);
        let loss_down = average_entropy_loss(&predictions_down, target);

        weights[i] += epsilon;

        let gradient = (loss_up - loss_down) / (2.0 * epsilon);
        weights[i] -= learning_rate * gradient;
    }

    model.set_weights(&weights);
}
