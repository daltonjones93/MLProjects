#include <iostream>
#include <cmath>
#include <vector>
#include <cstdlib>
#include <ctime>

using namespace std;

// Sigmoid activation function
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// Derivative of the sigmoid function
double sigmoid_derivative(double x) {
    return x * (1.0 - x);
}

class NeuralNetwork {
public:
    NeuralNetwork(int input_size, int hidden_size, int output_size) {
        input_size_ = input_size;
        hidden_size_ = hidden_size;
        output_size_ = output_size;

        // Initialize weights and biases randomly
        srand(time(0));
        weights_ih_.resize(hidden_size_, vector<double>(input_size_));
        weights_ho_.resize(output_size_, vector<double>(hidden_size_));
        biases_h_.resize(hidden_size_);
        biases_o_.resize(output_size_);

        for (int i = 0; i < hidden_size_; ++i) {
            for (int j = 0; j < input_size_; ++j) {
                weights_ih_[i][j] = (rand() % 100) / 100.0;  // Random values between 0 and 1
            }
            biases_h_[i] = (rand() % 100) / 100.0;
        }

        for (int i = 0; i < output_size_; ++i) {
            for (int j = 0; j < hidden_size_; ++j) {
                weights_ho_[i][j] = (rand() % 100) / 100.0;
            }
            biases_o_[i] = (rand() % 100) / 100.0;
        }
    }

    // Forward pass
    vector<double> predict(const vector<double>& input) {
        // Input to hidden
        vector<double> hidden(hidden_size_);
        for (int i = 0; i < hidden_size_; ++i) {
            hidden[i] = 0;
            for (int j = 0; j < input_size_; ++j) {
                hidden[i] += input[j] * weights_ih_[i][j];
            }
            hidden[i] = sigmoid(hidden[i] + biases_h_[i]);
        }

        // Hidden to output
        vector<double> output(output_size_);
        for (int i = 0; i < output_size_; ++i) {
            output[i] = 0;
            for (int j = 0; j < hidden_size_; ++j) {
                output[i] += hidden[j] * weights_ho_[i][j];
            }
            output[i] = sigmoid(output[i] + biases_o_[i]);
        }

        return output;
    }

    // Backward pass and weight update using gradient descent
    void train(const vector<vector<double> >& inputs, const vector<vector<double> >& targets, double learning_rate, int epochs) {
        for (int epoch = 0; epoch < epochs; ++epoch) {
            for (size_t i = 0; i < inputs.size(); ++i) {
                // Forward pass
                vector<double> input = inputs[i];
                vector<double> target = targets[i];

                // Input to hidden
                vector<double> hidden(hidden_size_);
                for (int j = 0; j < hidden_size_; ++j) {
                    hidden[j] = 0;
                    for (int k = 0; k < input_size_; ++k) {
                        hidden[j] += input[k] * weights_ih_[j][k];
                    }
                    hidden[j] = sigmoid(hidden[j] + biases_h_[j]);
                }

                // Hidden to output
                vector<double> output(output_size_);
                for (int j = 0; j < output_size_; ++j) {
                    output[j] = 0;
                    for (int k = 0; k < hidden_size_; ++k) {
                        output[j] += hidden[k] * weights_ho_[j][k];
                    }
                    output[j] = sigmoid(output[j] + biases_o_[j]);
                }

                // Backward pass
                // Calculate output layer errors and deltas
                vector<double> output_errors(output_size_);
                vector<double> output_deltas(output_size_);
                for (int j = 0; j < output_size_; ++j) {
                    output_errors[j] = target[j] - output[j];
                    output_deltas[j] = output_errors[j] * sigmoid_derivative(output[j]);
                }

                // Update hidden to output weights and biases
                for (int j = 0; j < output_size_; ++j) {
                    for (int k = 0; k < hidden_size_; ++k) {
                        weights_ho_[j][k] += learning_rate * output_deltas[j] * hidden[k];
                    }
                    biases_o_[j] += learning_rate * output_deltas[j];
                }

                // Calculate hidden layer errors and deltas
                vector<double> hidden_errors(hidden_size_);
                vector<double> hidden_deltas(hidden_size_);
                for (int j = 0; j < hidden_size_; ++j) {
                    hidden_errors[j] = 0;
                    for (int k = 0; k < output_size_; ++k) {
                        hidden_errors[j] += output_deltas[k] * weights_ho_[k][j];
                    }
                    hidden_deltas[j] = hidden_errors[j] * sigmoid_derivative(hidden[j]);
                }

                // Update input to hidden weights and biases
                for (int j = 0; j < hidden_size_; ++j) {
                    for (int k = 0; k < input_size_; ++k) {
                        weights_ih_[j][k] += learning_rate * hidden_deltas[j] * input[k];
                    }
                    biases_h_[j] += learning_rate * hidden_deltas[j];
                }
            }
        }
    }

private:
    int input_size_;
    int hidden_size_;
    int output_size_;

    vector<vector<double> > weights_ih_;  // Weights from input to hidden
    vector<vector<double> > weights_ho_;  // Weights from hidden to output
    vector<double> biases_h_;            // Biases for hidden layer
    vector<double> biases_o_;            // Biases for output layer
};

int main() {
    // Example usage
    // XOR problem
    vector<vector<double> > inputs = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<vector<double> > targets = {{0}, {1}, {1}, {0}};

    int input_size = 2;
    int hidden_size = 2;
    int output_size = 1;

    NeuralNetwork neural_network(input_size, hidden_size, output_size);

    // Train the neural network
    neural_network.train(inputs, targets, 0.1, 100000);

    // Test the trained neural network
    for (size_t i = 0; i < inputs.size(); ++i) {
        vector<double> input = inputs[i];
        vector<double> predicted_output = neural_network.predict(input);

        cout << "Input: {" << input[0] << ", " << input[1] << "} ";
        cout << "Predicted Output: " << predicted_output[0] << endl;
    }

    return 0;
}
