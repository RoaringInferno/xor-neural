#pragma once

#include "armadillo"
// using namespace arma;

namespace neural
{
    /**
     * @class layer
     * 
     * @brief A class that represents a layer in a neural network.
     * The layer class stores the weights and biases of its layer.
     */
    class layer
    {
    // Typedefs
    public:
        typedef unsigned int width_t;
    // Members
    private:
        /**
         * @brief The weights of the layer.
         * row count is output width, column count is input width.
         */
        arma::Mat<float> weights;
        /**
         * @brief The biases of the layer.
         * row count is output width.
         */
        arma::Col<float> biases;
    // Constructors
    public:
        /**
         * @brief Construct a new layer object
         * 
         * Initializes the weights with random values between -0.5 and 0.5
         * and the biases with zeros.
         */
        layer(const width_t& input_width, const width_t& output_width);
        ~layer();
    // Methods
    public:
        /**
         * @brief Evaluate the layer with the given input.
         * 
         * Returns the column vector result.
         */
        arma::Col<float> evaluate(const arma::Col<float>& input);

        arma::Mat<float>& get_weights();
        arma::Col<float>& get_biases();

        const arma::Mat<float>& get_weights() const;
        const arma::Col<float>& get_biases() const;

        void set_weights(const arma::Mat<float>& _weights);
        void set_biases(const arma::Col<float>& _biases);

        width_t get_output_width() const;
        width_t get_input_width() const;
    };
}