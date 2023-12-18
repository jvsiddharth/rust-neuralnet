use rand::Rng;

const E: f64 = std::f64::consts::E;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + E.powf(-x))
}

pub struct NeuralNetwork {
    input_nodes: usize,
    hidden_nodes: usize,
    output_nodes: usize,
    learning_rate: f64,
    weights_input_hidden: Vec<Vec<f64>>,
    weights_hidden_output: Vec<Vec<f64>>,
}

impl NeuralNetwork {
    pub fn new(
        input_nodes: usize,
        hidden_nodes: usize,
        output_nodes: usize,
        learning_rate: f64,
    ) -> Self {
        let mut rng = rand::thread_rng();

        let weights_input_hidden = (0..hidden_nodes)
            .map(|_| (0..input_nodes).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();

        let weights_hidden_output = (0..output_nodes)
            .map(|_| (0..hidden_nodes).map(|_| rng.gen::<f64>() - 0.5).collect())
            .collect();

        NeuralNetwork {
            input_nodes,
            hidden_nodes,
            output_nodes,
            learning_rate,
            weights_input_hidden,
            weights_hidden_output,
        }
    }

    pub fn train(&mut self, inputs: &[f64], target: &[f64]) {
        //forward pass
        let mut hidden_outputs = vec![0.0; self.hidden_nodes];
        for i in 0..self.hidden_nodes {
            let mut sum = 0.0;
            for j in 0..self.input_nodes {
                sum += self.weights_input_hidden[i][j] * inputs[j];
            }
            hidden_outputs[i] = sigmoid(sum);
        }

        let mut final_outputs = vec![0.0; self.output_nodes];
        for i in 0..self.output_nodes {
            let mut sum = 0.0;
            for j in 0..self.hidden_nodes {
                sum += self.weights_hidden_output[i][j] * hidden_outputs[j];
            }
            final_outputs[i] = sigmoid(sum);
        }

        // backpropogation
        let mut output_errors = vec![0.0; self.output_nodes];
        for i in 0..self.output_nodes {
            output_errors[i] = target[i] - final_outputs[i];
        }

        let mut hidden_errors = vec![0.0; self.hidden_nodes];
        for i in 0..self.hidden_nodes {
            let mut error = 0.0;
            for j in 0..self.output_nodes {
                error += self.weights_hidden_output[j][i] * output_errors[j];
            }
            hidden_errors[i] = error;
        }

        //update weights_hidden_output

        for i in 0..self.output_nodes {
            for j in 0..self.hidden_nodes {
                self.weights_hidden_output[i][j] += self.learning_rate
                    * output_errors[i]
                    * final_outputs[i]
                    * (1.0 - final_outputs[i])
                    * hidden_outputs[j];
            }
        }

        //update weights_input_hidden
        for i in 0..self.hidden_nodes {
            for j in 0..self.input_nodes {
                self.weights_input_hidden[i][j] += self.learning_rate
                    * hidden_errors[i]
                    * hidden_outputs[i]
                    * (1.0 - hidden_outputs[i])
                    * inputs[j];
            }
        }
    }

    pub fn query(&self, inputs: &[f64]) -> Vec<f64> {
        let mut hidden_outputs = vec![0.0; self.hidden_nodes];
        for i in 0..self.hidden_nodes {
            let mut sum = 0.0;
            for j in 0..self.input_nodes {
                sum += self.weights_input_hidden[i][j] * inputs[j];
            }
            hidden_outputs[i] = sigmoid(sum);
        }

        let mut final_outputs = vec![0.0; self.output_nodes];
        for i in 0..self.output_nodes {
            let mut sum = 0.0;
            for j in 0..self.hidden_nodes {
                sum += self.weights_hidden_output[i][j] * hidden_outputs[j];
            }
            final_outputs[i] = sigmoid(sum);
        }
        final_outputs
    }
}
