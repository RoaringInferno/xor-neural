#pragma once

#include "matrix.hpp"
#include "activation.hpp"
#include "cost.hpp"

#include <random>
#include <thread>

namespace xneur
{
    template <size_t i, size_t o>
    class layer
    {
    public: // Constants
        static const size_t input_size = i;
        static const size_t output_size = o;
    
        typedef matrix<o, i> weight_matrix;
        typedef colvec<i> input_vector_t;
        typedef colvec<o> output_vector_t;
        typedef output_vector_t bias_vector;
    private: // Members
        weight_matrix weights;
        bias_vector biases;
    public: // Settings
        const cost_function& COST{cost::mean_squared_error};
    private: // Helper functions
        output_vector_t apply_weights(const input_vector_t &input) const { return weights * input; }
        output_vector_t apply_biases(const output_vector_t &output) const { return output + biases; }
        output_vector_t apply_activation(const output_vector_t &output) const
        {
            output_vector_t result;
            std::transform(output.begin(), output.end(), result.begin(), [](float x) { return sigmoid(x); });
            return result;
        }
    public: // Forward propagation
        output_vector_t evaluate(const input_vector_t &input) const { return apply_activation(apply_biases(apply_weights(input))); }
        output_vector_t operator()(const input_vector_t &input) const { return evaluate(input); }
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
        jacobian endpoint_backpropagate(
			const input_vector_t &input,
			const output_vector_t &output,
			const output_vector_t &expected
		) const
        {
            /*
            dL_dY = derivative of cost function
            dY_dA = sigmoid'(A) = sigmoid(A) * (1 - sigmoid(A))
            dA_dW = input
            dA_dB = 1
            dL_dW = dL_dY * dY_dA * dA_dW = dL_dY * dY_dA * input = dL_dB * input
            dL_dB = dL_dY * dY_dA * dA_dB = dL_dY * dY_dA
             */
        
            jacobian result;

            auto compute_output_node_gradient = [&](size_t output_index) {
                result.biases_gradient[output_index] =
                    COST.derivative(output[output_index], expected[output_index]) // dL_dY
                    * output[output_index] * (1.0f - output[output_index]); // dY_dA
                    //* 1.0f; // dA_dB
                for (size_t input_index = 0; input_index < input_size; input_index++)
                {
                    result.weight_gradients[output_index][input_index] =
                        result.biases_gradient[output_index] // dL_dY * dY_dA
                        * input[input_index]; // dA_dW
                }
            };

            for (size_t output_index = 0; output_index < output_size; output_index++)
            {
                compute_output_node_gradient(output_index);
            }
            return result;
        }
		
		/**
         * @brief Backpropagate the error through the layer, assuming the layer is not the output layer.
         * 
         * @param input The input to the layer
         * @param output The output of the layer
         * @param delta The transposed weight matrix of the next layer multiplied by the biases gradient of the next layer
		 * @return jacobian The jacobian of the layer
         */
		jacobian chained_backpropagate(
			const input_vector_t &input,
			const output_vector_t &output,
            const bias_vector &delta
        ) const
        {
			/*
            dL_dY = derivative of cost function
            dY_dA = sigmoid'(A) = sigmoid(A) * (1 - sigmoid(A))
            dA_dW = input
            dA_dB = 1
            dL_dW = dL_dY * dY_dA * dA_dW = dL_dY * dY_dA * input = dL_dB * input
            dL_dB = dL_dY * dY_dA * dA_dB = dL_dY * dY_dA
             */

            jacobian result;

            auto compute_output_node_gradient = [&](size_t output_index){
                result.biases_gradient[output_index] *=
                    output[output_index] * (1.0f - output[output_index]); // dY_dA
                    //* 1.0f; // dA_dB
                for (size_t input_index = 0; input_index < input_size; input_index++)
                {
                    result.weight_gradients[output_index][input_index] =
                        result.biases_gradient[output_index] // dL_dY * dY_dA
                        * input[input_index]; // dA_dW
                }
            };

            // Get the bias gradient(delta)
            result.biases_gradient = delta; // fire together, wire together

            for (size_t output_index = 0; output_index < output_size; output_index++)
            {
                compute_output_node_gradient(output_index);
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