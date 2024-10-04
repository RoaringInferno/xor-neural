#pragma once

#include "layer.hpp"

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
    public:
        layer::width_t get_input_width() const;
        layer::width_t get_output_width() const;

        Col<float> evaluate(const Col<float>& input);
    };
}