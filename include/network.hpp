#pragma once

#include "layer.hpp"

#include <vector>

namespace neural
{
    /**
     * @class NeuralNetwork
     * @brief A class that represents a neural network
     * 
     * Terms:
     * - Width: The number of neurons in a layer
     * - Depth: The number of layers in the network
     */
    class network
    {
    // Typedefs
    public:
        typedef unsigned short depth_t;
    // Structs
    public:
        /**
         * @struct training_data_set_t
         * 
         * @brief A struct that represents the training data for a network.
         * 
         * Simply a data wrapper.
         */
        struct training_data_set_t
        {
            std::vector<arma::Col<float>> inputs;
            std::vector<arma::Col<float>> expected_outputs;
        };
    private:
        /**
         * @struct gradient_t
         * 
         * @brief A struct that represents the gradient of a network.
         * 
         * Simply a data wrapper.
         */
        struct gradient_t
        {
            arma::Mat<float>* d_weights;
            arma::Col<float>* d_biases;

            gradient_t(const depth_t& depth);
            ~gradient_t();
        };
    // Members
    private:
        depth_t depth;
        layer* layers;
    // Constructors
    public:
        network();
        network(const depth_t& depth, const layer::width_t* widths);
        ~network();
    // Methods
    private:
        gradient_t backpropagate(const training_data_set_t& training_data);
    public:
        layer::width_t get_input_width() const;
        layer::width_t get_output_width() const;

        arma::Col<float> evaluate(const arma::Col<float>& input);
        float cost(const training_data_set_t& training_data);

        void train(const training_data_set_t& training_data, const float& learning_rate);
    };
}