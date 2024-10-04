#pragma once

#include "armadillo"
using namespace arma;

namespace neural
{
    class layer
    {
    // Typedefs
    public:
        typedef unsigned int width_t;
    // Members
    private:
        Mat<float> weights;
        Col<float> biases;
    // Constructors
    public:
        layer();
        layer(const width_t& input_width, const width_t& output_width);
        ~layer();
    // Methods
    private:
        width_t get_output_width() const;
        width_t get_input_width() const;
    public:
        Col<float> evaluate(const Col<float>& input);
    };
}