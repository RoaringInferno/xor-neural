#pragma once

#include "layer.hpp"

namespace xneur
{
    struct training_hyperparameters
    {
        size_t epochs{100};
        float learning_rate{0.01f};
    };

    /// @brief Simple neural network class. Assumes more than 1 hidden layer.
    /// @tparam I Input size
    /// @tparam O Output size
    /// @tparam H Hidden layer size
    /// @tparam C Hidden layer count
    template<size_t I, size_t O, size_t H, size_t C>
    class network
    {
        /**
         * A neural network with a rectangular hidden layer structure.
         * Terms:
         * - Input layer: The first hidden layer of the network (special because it connects an input size to a hidden size)
         * - Hidden layer: A layer that connects a hidden size to a hidden size
         * - Output layer: The last hidden layer of the network (special because it connects a hidden size to an output size, and because it is the last layer)
         * 
         * All indexes are 0-based, shifted down by 1 (i.e. the 5th hidden layer is at index 4, not 5).
         *  This is because the first hidden layer is classified as an "input layer", and so is not present in indexing.
         *  All data structures are to be indexed in this way. The only data structure that is not indexed in this way by defualt is the forward_propagation_result struct.
         *  However, the forward_propagation_result struct has an operator[] that is indexed in this way.
         * 
         * true_hidden_count reflects the number of hidden layers in the hidden_layers array, not including the input layer.
         * last_hidden_index reflects the index of the last hidden layer in the hidden_layers array, not including the input layer.
         * These should be used to index all data structures.
         */
    public:
        const size_t input_size = I; // The size of the input layer
        const size_t output_size = O; // The size of the output layer
        const size_t hidden_size = H; // The size of the hidden layers
        const size_t hidden_count = C; // The number of hidden layers in the matrix
        const size_t true_hidden_count = hidden_count - 1; // The number of hidden layer objects in the array. The first layer is the input layer, not a hidden layer.
        const size_t last_hidden_index = true_hidden_count - 1; // The index of the last hidden layer in the array
    public:
        typedef typename layer<I, H> input_layer;
        typedef typename layer<H, H> hidden_layer;
        typedef typename layer<H, O> output_layer;

        typedef typename input_layer::input_vector input_vector;
        typedef typename output_layer::output_vector output_vector;
        typedef typename hidden_layer::output_vector hidden_vector;
    
        typedef typename input_layer::jacobian input_jacobian;
        typedef typename output_layer::jacobian output_jacobian;
        typedef typename hidden_layer::jacobian hidden_jacobian;
    private:
        input_layer input_layer; // The input layer
        std::array<hidden_layer, true_hidden_count> hidden_layers; // The hidden layers. The first layer is the input layer, not a hidden layer. (index shifted down by 1)
        output_layer output_layer; // The output layer
    public:
        struct jacobian
        {
            input_jacobian input_layer_jacobian; // The jacobian of the input layer
            std::array<hidden_jacobian, true_hidden_count> hidden_layers_jacobian; // The jacobian of the hidden layers. The first layer is the input layer, not a hidden layer. (index shifted down by 1)
            output_jacobian output_layer_jacobian; // The jacobian of the output layer
        };

        struct forward_propagation_result
        /**
         * WARNING
         * 
         * This is the only data structure that is not indexed in the same way as the rest of the data structures.
         */
        {
            input_vector input; // The input to the network
            std::array<hidden_vector, hidden_count> hidden_layers_output; // The output of the hidden layers(including the input layer; index unshifted)
            output_vector output_layer_output; // The output of the output layer

            forward_propagation_result(const input_vector &input) : input(input) {}

            hidden_vector &input_layer_output() { return hidden_layers_output[0]; } // The input layer's output
            hidden_vector &operator[](size_t index) { return hidden_layers_output[index + 1]; } // The hidden layers. The first layer is the input layer, not a hidden layer, so the index is increased by 1 to account for that.
        };
    public:
        output_vector evaluate(const input_vector &input) const
        {
            auto hidden_output = input_layer(input);
            for (const auto &layer : hidden_layers)
            {
                hidden_output = layer(hidden_output);
            }
            return output_layer(hidden_output);
        }

        forward_propagation_result forward_propagate(const input_vector &input) const
        {
            forward_propagation_result result(input);
            result.input_layer_output() = input_layer(input); // The input layer
            for (size_t output_index = 0; output_index < true_hidden_count; output_index++) // The hidden layers
            {
                size_t input_index = output_index - 1;
                result[output_index] = hidden_layers[output_index](result[input_index]);
            }
            result.output_layer_output = output_layer(result[last_hidden_index]); // The output layer
            return result;
        }
    
        jacobian back_propagate(const input_vector &input, const output_vector &expected) const
        {
            /**
             * Because every hidden layer requires the next layer's weight and bias gradient information to backpropagate, we must compute the last hidden layer before beginning the loop.
             * The forward propagation result is stored as:
             * - input
             * - hidden_layers_output<true_hidden_count>
             * - output_layer_output
             * Because the loop requires the current, next, and previous layer's information, we must compute as follows:
             * - The output layer (using expected and the last hidden layer)
             * - The last hidden layer (using the output layer and the second-to-last hidden layer)
             * - The rest of the hidden layers<true_hidden_count - 1> (using the next hidden layer and the previous hidden layer).
             *  - Start from the second-to-last hidden layer and work backwards to the second hidden layer (at index 0; The first is the input layer, remember).
             * - The input layer (using the input and the first hidden layer)
             */

            jacobian gradient;
            auto forward_propagation = forward_propagate(input);
            gradient.output_layer_jacobian = output_layer.endpoint_backpropagate( // The output layer
                forward_propagation[last_hidden_index], // The last hidden layer
                forward_propagation.output_layer_output, // The output layer
                expected // The expected output
            );
            gradient.hidden_layers_jacobian[last_hidden_index] = hidden_layers[last_hidden_index].chained_backpropagate( // The last hidden layer
                forward_propagation[last_hidden_index-1], // The second-to-last hidden layer
                forward_propagation[last_hidden_index], // The last hidden layer
                output_layer.get_weights().transpose_dot(gradient.output_layer_jacobian.biases_gradient) // The output layer
            );

            for (size_t index = last_hidden_index - 1; index < last_hidden_index; index--) // The rest of the hidden layers
            {
                gradient.hidden_layers_jacobian[index] = hidden_layers[index].chained_backpropagate(
                    forward_propagation[index - 1], // index - 1
                    forward_propagation[index], // index
                    hidden_layer[index + 1].weights.transpose_dot(gradient.hidden_layers_jacobian[index + 1].biases_gradient); // index + 1
                );
            }
            gradient.input_layer_jacobian = input_layer.chained_backpropagate(
                forward_propagation.input, // -2
                forward_propagation.input_layer_output(), // -1
                hidden_layers[0].weights.transpose_dot(gradient.hidden_layers_jacobian[0].biases_gradient) // 0
            );
            return gradient;
        }

        void train(
            const input_vector &input,
            const output_vector &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                auto gradient = back_propagate(input, expected);
                input_layer.weights -= gradient.input_layer_jacobian.weight_gradients * hyperparameters.learning_rate;
                input_layer.biases -= gradient.input_layer_jacobian.biases_gradient * hyperparameters.learning_rate;
                for (size_t j = 0; j < true_hidden_count; j++)
                {
                    hidden_layers[j].weights -= gradient.hidden_layers_jacobian[j].weight_gradients * hyperparameters.learning_rate;
                    hidden_layers[j].biases -= gradient.hidden_layers_jacobian[j].biases_gradient * hyperparameters.learning_rate;
                }
                output_layer.weights -= gradient.output_layer_jacobian.weight_gradients * hyperparameters.learning_rate;
                output_layer.biases -= gradient.output_layer_jacobian.biases_gradient * hyperparameters.learning_rate;
            }
        }
    };

    template<size_t I, size_t O, size_t H, size_t C>
    class network<I, O, H, 1> : public network<I, O, H, C>
    {
    public:
        const size_t input_size = I;
        const size_t output_size = O;
        const size_t hidden_size = H;
    public:
        typedef typename layer<I, H> input_layer;
        typedef typename layer<H, O> output_layer;

        typedef typename input_layer::input_vector input_vector;
        typedef typename output_layer::output_vector output_vector;
        typedef typename hidden_layer::output_vector hidden_vector;

        typedef typename input_layer::jacobian input_jacobian;
        typedef typename output_layer::jacobian output_jacobian;
    private:
        input_layer input_layer;
        output_layer output_layer;
    public:
        struct jacobian
        {
            input_jacobian input_layer_jacobian;
            output_jacobian output_layer_jacobian;
        };

        struct forward_propagation_result
        /**
         * WARNING
         * 
         * This is the only data structure that is not indexed in the same way as the rest of the data structures.
         */
        {
            typename input_vector input;
            typename hidden_vector hidden_layer_output;
            typename output_vector output_layer_output;

            forward_propagation_result(const input_vector &input) : input(input) {}
        };
    public:
        output_vector evaluate(const input_vector &input) const
        {
            return output_layer(input_layer(input));
        }

        forward_propagation_result forward_propagate(const input_vector &input) const
        {
            forward_propagation_result result(input);
            result.input_layer_output = input_layer(input);
            result.output_layer_output = output_layer(result.input_layer_output);
            return result;
        }

        jacobian back_propagate(const input_vector &input, const output_vector &expected) const
        {
            jacobian gradient;
            auto forward_propagation = forward_propagate(input);
            gradient.output_layer_jacobian = output_layer.endpoint_backpropagate(
                forward_propagation.hidden_layers_output[last_hidden_index], // hidden
                forward_propagation.output_layer_output, // output
                expected // expected
            );

            gradient.input_layer_jacobian = input_layer.chained_backpropagate(
                forward_propagation.input, // input
                forward_propagation.hidden_layer_output, // hidden
                output_layer.weights.transpose_dot(gradient.output_layer_jacobian.biases_gradient) // output
            );
            return gradient;
        }

        void train(
            const input_vector &input,
            const output_vector &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                auto gradient = back_propagate(input, expected);
                input_layer.weights -= gradient.input_layer_jacobian.weight_gradients * hyperparameters.learning_rate;
                input_layer.biases -= gradient.input_layer_jacobian.biases_gradient * hyperparameters.learning_rate;
                output_layer.weights -= gradient.output_layer_jacobian.weight_gradients * hyperparameters.learning_rate;
                output_layer.biases -= gradient.output_layer_jacobian.biases_gradient * hyperparameters.learning_rate;
            }
        }
    };

    template<size_t I, size_t O>
    class single_layer_network : public layer<I, O>
    {
    public:
        output_vector forward_propagate(const input_vector &input) const
        {
            return this->evaluate(input);
        }

        jacobian back_propagate(const input_vector &input, const output_vector &expected) const
        {
            return this->endpoint_backpropagate(input, this->evaluate(input), expected);
        }

        void train(
            const input_vector &input,
            const output_vector &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                auto gradient = this->back_propagate(input, this->forward_propagate(input), expected);
                this->weights -= gradient.weight_gradients * hyperparameters.learning_rate;
                this->biases -= gradient.biases_gradient * hyperparameters.learning_rate;
            }
        }
    };
} // namespace xneur