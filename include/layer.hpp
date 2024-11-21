#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "cost.hpp"

#include <thread>

namespace xneur
{
    template <size_t i, size_t o>
    class layer
    {
    public: // Constants
        static const size_t input_size = input;
        static const size_t output_size = output;
    
        typedef matrix<o, i> weight_matrix;
        typedef colvec<i> input_vector;
        typedef colvec<o> output_vector;
        typedef output_vector bias_vector;
    private: // Members
        weight_matrix weights;
        bias_vector biases;
    public: // Settings
        const cost_function& COST{cost::mean_squared_error};
        const bool BACKPROPAGATION_COMPUTE_NODE_THREADED{false};
    private: // Helper functions
        output_vector apply_weights(const input_vector &input) const { return weights * input; }
        output_vector apply_biases(const output_vector &output) const { return output + biases; }
        output_vector apply_activation(const output_vector &output) const
        {
            output_vector result;
            std::transform(output.begin(), output.end(), result.begin(), [](float x) { return sigmoid(x); });
            return result;
        }
    public: // Forward propagation
        output_vector forward_propagate(const input_vector &input) const { return apply_activation(apply_biases(apply_weights(input))); }
        output_vector operator()(const input_vector &input) const { return forward_propagate(input); }
    public: // Backpropagation
        struct jacobian
        {
            weight_matrix weight_gradients;
            bias_vector biases_gradient;
        };

        /**
         * @brief Backpropagate the error through the layer, as if the layer was the output layer
         * 
         * @param input The input to the layer
         * @param output The output of the layer
         * @param expected The expected output of the layer
         * @return jacobian The jacobian of the layer
         */
        jacobian backpropagate(const input_vector &input, const output_vector &output, const output_vector &expected) const
        {
            /*
            dL_dY = derivative of cost function
            dY_dA = sigmoid'(A) = sigmoid(A) * (1 - sigmoid(A))
            dA_dW = input
            dA_dB = 1
            dL_dW = dL_dY * dY_dA * dA_dW = dL_dY * dY_dA * input = dL_dB * input
            dL_dB = dL_dY * dY_dA * dA_dB = dL_dY * dY_dA
             */

            auto compute_output_node_gradient = [&](size_t output_index){
                result.biases_gradient[output_index] =
                    COST.derivative(output[output_index], expected[output_index]) // dL_dY
                    * sigmoid(output[output_index]) * (1.0f - sigmoid(output[output_index])) // dY_dA
                    * 1.0f; // dA_dB
                for (size_t j = 0; j < input_size; j++)
                {
                    result.weight_gradients[output_index][j] =
                        result.biases_gradient[output_index] // dL_dY * dY_dA
                        * input[j]; // dA_dW
                }
            }

            jacobian result;
            if (BACKPROPAGATION_COMPUTE_NODE_THREADED)
            {
                std::vector<std::thread> threads;
                for (size_t i = 0; i < output_vector; i++)
                {
                    threads.push_back(std::thread(compute_output_node_gradient, i));
                }
                for (auto &thread : threads)
                {
                    thread.join();
                }
            }
            else
            {
                for (size_t i = 0; i < output_vector; i++)
                {
                    compute_output_node_gradient(i);
                }
            }
            return result;
        }
    public: // Accessors
        const weight_matrix &get_weights() const { return weights; }
        const bias_vector &get_biases() const { return biases; }
        void set_weights(const weight_matrix &w) { weights = w; }
        void set_biases(const bias_vector &b) { biases = b; }
    public: // Initialization
        enum INIT_STATE : bool
        {
            RANDOM = true,
            ZERO = false
        };

        void randomize()
        {
            std::random_device rd;
            std::mt19937 gen(rd());
            std::normal_distribution<float> dist(0.0f, 1.0f);
            std::generate(weights.cell_begin(), weights.cell_end(), [&dist, &gen]() { return dist(gen); });
            std::generate(biases.begin(), biases.end(), [&dist, &gen]() { return dist(gen); });
        }
        void zero_out()
        {
            std::fill(weights.cell_begin(), weights.cell_end(), 0.0f);
            std::fill(biases.begin(), biases.end(), 0.0f);
        }
        void reset(INIT_STATE state = RANDOM)
        {
            if (state == RANDOM)
            {
                randomize();
            }
            else
            {
                zero_out();
            }
        }
    public: // Constructors
        layer(INIT_STATE state = RANDOM) { reset(state); }
        layer(const weight_matrix &w, const bias_vector &b) : weights(w), biases(b) {}
        layer(const layer &l) : weights(l.weights), biases(l.biases) {}
        layer(layer &&l) : weights(std::move(l.weights)), biases(std::move(l.biases)) {}
    };
} // namespace xneur