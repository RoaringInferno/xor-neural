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
        static const size_t input_size = I; // The size of the input layer
        static const size_t output_size = O; // The size of the output layer
        static const size_t hidden_size = H; // The size of the hidden layers
        static const size_t hidden_count = C; // The number of hidden layers in the matrix
        static const size_t true_hidden_count = hidden_count - 1; // The number of hidden layer objects in the array. The first layer is the input layer, not a hidden layer.
        static const size_t last_hidden_index = true_hidden_count - 1; // The index of the last hidden layer in the array
    public:
        typedef layer<I, H> input_layer_t;
        typedef layer<H, H> hidden_layer_t;
        typedef layer<H, O> output_layer_t;

        typedef typename input_layer_t::input_vector_t input_vector_t;
        typedef typename output_layer_t::output_vector_t output_vector_t;
        typedef typename hidden_layer_t::output_vector_t hidden_vector_t;
    
        typedef typename input_layer_t::jacobian input_jacobian_t;
        typedef typename output_layer_t::jacobian output_jacobian_t;
        typedef typename hidden_layer_t::jacobian hidden_jacobian_t;
    private:
        input_layer_t input_layer; // The input layer
        std::array<hidden_layer_t, true_hidden_count> hidden_layers; // The hidden layers. The first layer is the input layer, not a hidden layer. (index shifted down by 1)
        output_layer_t output_layer; // The output layer
    public:
        struct jacobian
        {
            input_jacobian_t input_layer_jacobian; // The jacobian of the input layer
            std::array<hidden_jacobian_t, true_hidden_count> hidden_layers_jacobian; // The jacobian of the hidden layers. The first layer is the input layer, not a hidden layer. (index shifted down by 1)
            output_jacobian_t output_layer_jacobian; // The jacobian of the output layer
        };

        struct forward_propagation_result
        /**
         * WARNING
         * 
         * This is the only data structure that is not indexed in the same way as the rest of the data structures.
         */
        {
            input_vector_t input; // The input to the network
            std::array<hidden_vector_t, hidden_count> hidden_layers_output; // The output of the hidden layers(including the input layer; index unshifted)
            output_vector_t output_layer_output; // The output of the output layer

            hidden_vector_t &operator[](size_t index) { return hidden_layers_output[index + 1]; } // The hidden layers. The first layer is the input layer, not a hidden layer, so the index is increased by 1 to account for that.
            hidden_vector_t &input_layer_output() { return hidden_layers_output[0]; } // The input layer's output

            forward_propagation_result(const input_vector_t &input) :
                input(input)
            {
                this->input_layer_output() = input_layer(input); // The input layer
                for (size_t output_index = 0; output_index < true_hidden_count; output_index++) // The hidden layers
                {
                    size_t input_index = output_index - 1;
                    (*this)[output_index] = hidden_layers[output_index]((*this)[input_index]);
                }
                this->output_layer_output = output_layer((*this)[last_hidden_index]); // The output layer
            }
        };

        struct training_data_t
        {
            input_vector_t input;
            output_vector_t expected;
        };
    public:
        output_vector_t evaluate(const input_vector_t &input) const
        {
            auto hidden_output = input_layer(input);
            for (const auto &layer : hidden_layers)
            {
                hidden_output = layer(hidden_output);
            }
            return output_layer(hidden_output);
        }

        forward_propagation_result forward_propagate(const input_vector_t &input) const
        {
            forward_propagation_result result(input);
            /*
            result.input_layer_output() = input_layer(input); // The input layer
            for (size_t output_index = 0; output_index < true_hidden_count; output_index++) // The hidden layers
            {
                size_t input_index = output_index - 1;
                result[output_index] = hidden_layers[output_index](result[input_index]);
            }
            result.output_layer_output = output_layer(result[last_hidden_index]); // The output layer
            */
            return result;
        }
    
        jacobian back_propagate(const input_vector_t &input, const output_vector_t &expected) const
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
                    hidden_layers[index + 1].weights.transpose_dot(gradient.hidden_layers_jacobian[index + 1].biases_gradient) // index + 1
                );
            }
            gradient.input_layer_jacobian = input_layer.chained_backpropagate(
                forward_propagation.input, // -2
                forward_propagation.input_layer_output(), // -1
                hidden_layers[0].weights.transpose_dot(gradient.hidden_layers_jacobian[0].biases_gradient) // 0
            );
            return gradient;
        }

        jacobian back_propagate(const training_data_t &training_data) const
        {
            return back_propagate(training_data.input, training_data.expected);
        }

        void train(
            const input_vector_t &input,
            const output_vector_t &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters()
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                jacobian gradient = back_propagate(input, expected);
                input_layer.set_weights(input_layer.get_weights() - gradient.input_layer_jacobian.weight_gradients * hyperparameters.learning_rate);
                input_layer.set_biases(input_layer.get_biases() - gradient.input_layer_jacobian.biases_gradient * hyperparameters.learning_rate);
                for (size_t j = 0; j < true_hidden_count; j++)
                {
                    hidden_layers[j].set_weights(hidden_layers[j].get_weights() - gradient.hidden_layers_jacobian[j].weight_gradients * hyperparameters.learning_rate);
                    hidden_layers[j].set_biases(hidden_layers[j].get_biases() - gradient.hidden_layers_jacobian[j].biases_gradient * hyperparameters.learning_rate);
                }
                output_layer.set_weights(output_layer.get_weights() - gradient.output_layer_jacobian.weight_gradients * hyperparameters.learning_rate);
                output_layer.set_biases(output_layer.get_biases() - gradient.output_layer_jacobian.biases_gradient * hyperparameters.learning_rate);
            }
        }

        void train(
            const training_data_t &training_data,
            const training_hyperparameters &hyperparameters = training_hyperparameters()
        )
        {
            train(training_data.input, training_data.expected, hyperparameters);
        }
    public:
        network() :
            input_layer(input_layer_t::INIT_STATE::RANDOM),
            hidden_layers(),
            output_layer(output_layer_t::INIT_STATE::RANDOM)
        {
            for (auto &layer : hidden_layers)
            {
                layer.randomize();
            }
        };

        void reset()
        {
            input_layer.randomize();
            for (auto &layer : hidden_layers)
            {
                layer.randomize();
            }
            output_layer.randomize();
        }
    };

    template<size_t I, size_t O, size_t H>
    class network<I, O, H, 1>
    {
    public:
        static const size_t input_size = I;
        static const size_t output_size = O;
        static const size_t hidden_size = H;
    public:
        typedef layer<I, H> input_layer_t;
        typedef layer<H, O> output_layer_t;

        typedef typename input_layer_t::input_vector_t input_vector_t;
        typedef typename input_layer_t::output_vector_t hidden_vector_t;
        typedef typename output_layer_t::output_vector_t output_vector_t;

        typedef typename input_layer_t::jacobian input_jacobian_t;
        typedef typename output_layer_t::jacobian output_jacobian_t;
    private:
        input_layer_t input_layer;
        output_layer_t output_layer;
    public:
        struct jacobian
        {
            input_jacobian_t input_layer_jacobian;
            output_jacobian_t output_layer_jacobian;
        };

        struct forward_propagation_result
        /**
         * WARNING
         * 
         * This is the only data structure that is not indexed in the same way as the rest of the data structures.
         */
        {
            input_vector_t input;
            hidden_vector_t hidden_layer_output;
            output_vector_t output_layer_output;

            forward_propagation_result(const input_vector_t &input) :
                input(input),
                hidden_layer_output(input_layer(input))
            {
                output_layer_output = output_layer_t(hidden_layer_output);
            }

            hidden_vector_t &input_layer_output() { return hidden_layer_output; }
        };

        struct training_data_t
        {
            input_vector_t input;
            output_vector_t expected;
        };
    public:
        output_vector_t evaluate(const input_vector_t &input) const
        {
            return output_layer(input_layer(input));
        }

        forward_propagation_result forward_propagate(const input_vector_t &input) const
        {
            forward_propagation_result result(input);
            //result.input_layer_output() = input_layer(input);
            //result.output_layer_output = output_layer_t(result.input_layer_output);
            return result;
        }

        jacobian back_propagate(const input_vector_t &input, const output_vector_t &expected) const
        {
            jacobian gradient;
            auto forward_propagation = forward_propagate(input);
            gradient.output_layer_jacobian = output_layer.endpoint_backpropagate(
                forward_propagation.hidden_layer_output, // hidden
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

        jacobian back_propagate(const training_data_t &training_data) const
        {
            return back_propagate(training_data.input, training_data.expected);
        }

        void train(
            const input_vector_t &input,
            const output_vector_t &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                jacobian gradient = back_propagate(input, expected);
                input_layer.set_weights(input_layer.get_weights() - gradient.input_layer_jacobian.weight_gradients * hyperparameters.learning_rate);
                input_layer.set_biases(input_layer.get_biases() - gradient.input_layer_jacobian.biases_gradient * hyperparameters.learning_rate);
                output_layer.set_weights(output_layer.get_weights() - gradient.output_layer_jacobian.weight_gradients * hyperparameters.learning_rate);
                output_layer.set_biases(output_layer.get_biases() - gradient.output_layer_jacobian.biases_gradient * hyperparameters.learning_rate);
            }
        }

        void train(
            const training_data_t &training_data,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            train(training_data.input, training_data.expected, hyperparameters);
        }
    
    public:
        network() :
            input_layer(input_layer_t::INIT_STATE::RANDOM),
            output_layer(output_layer_t::INIT_STATE::RANDOM)
        {
        };

        void reset()
        {
            input_layer.randomize();
            output_layer.randomize();
        }
    };

    template<size_t I, size_t O>
    class single_layer_network
    {
    public:
        typedef layer<I, O> output_layer_t;

        typedef typename output_layer_t::input_vector_t input_vector_t;
        typedef typename output_layer_t::output_vector_t output_vector_t;

        typedef typename output_layer_t::jacobian jacobian;
    private:
        output_layer_t output_layer;
    public:

        struct forward_propagation_result
        {
            input_vector_t input;
            output_vector_t output;

            forward_propagation_result(const input_vector_t &input) : input(input), output(output_layer(input)) {}
        };

        struct training_data_t
        {
            input_vector_t input;
            output_vector_t expected;
        };
    public:
        output_vector_t evaluate(const input_vector_t &input) const
        {
            return output_layer(input);
        }

        forward_propagation_result forward_propagate(const input_vector_t &input) const
        {
            forward_propagation_result result(input);
            //result.output = output_layer(input);
            return result;
        }

        jacobian back_propagate(const input_vector_t &input, const output_vector_t &expected) const
        {
            jacobian gradient;
            forward_propagation_result forward_propagation = forward_propagate(input);
            gradient = output_layer.endpoint_backpropagate(
                forward_propagation.input,
                forward_propagation.output,
                expected
            );
            return gradient;
        }

        jacobian back_propagate(const training_data_t &training_data) const
        {
            return back_propagate(training_data.input, training_data.expected);
        }

        void train(
            const input_vector_t &input,
            const output_vector_t &expected,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            for (size_t i = 0; i < hyperparameters.epochs; i++)
            {
                jacobian gradient = back_propagate(input, expected);
                output_layer.set_weights(output_layer.get_weights() - gradient.output_layer_jacobian.weight_gradients * hyperparameters.learning_rate);
                output_layer.set_biases(output_layer.get_biases() - gradient.output_layer_jacobian.biases_gradient * hyperparameters.learning_rate);
            }
        }

        void train(
            const training_data_t &training_data,
            const training_hyperparameters &hyperparameters = training_hyperparameters{}
        )
        {
            train(training_data.input, training_data.expected, hyperparameters);
        }
    public:
        single_layer_network() :
            output_layer(output_layer_t::INIT_STATE::RANDOM)
        {
        };

        void reset()
        {
            output_layer.randomize();
        }
    };
} // namespace xneur