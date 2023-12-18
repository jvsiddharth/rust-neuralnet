mod neural_network;

use neural_network::NeuralNetwork;

fn main() {
    let input_nodes = 3;
    let hidden_nodes = 3;
    let output_nodes = 3;
    let learning_rate = 0.3;

    let mut neural_network =
        NeuralNetwork::new(input_nodes, hidden_nodes, output_nodes, learning_rate);

    // example training data
    let training_data = vec![(vec![0.5, 0.2, 0.1], vec![0.01, 0.99, 0.01])];

    for _ in 0..1000 {
        for (inputs, targets) in &training_data {
            neural_network.train(inputs, targets);
        }
    }

    let inputs_to_query = &[0.5, 0.2, 0.1];
    let output = neural_network.query(inputs_to_query);
    println!("Output: {:?}", output);
}
