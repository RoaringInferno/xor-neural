#include "layer.hpp"

#include "sigmoid.hpp"

neural::layer::layer(const width_t &input_width, const width_t &output_width) :
    weights(
        // Random values between -0.5 and 0.5
        arma::Mat<float>(
            output_width,
            input_width,
            arma::fill::randu // Random values between 0 and 1
        )
        -
        arma::Mat<float>(
            output_width,
            input_width,
            arma::fill::value(0.5f) // Subtract 0.5
        )
    ),
    biases(
        output_width,
        arma::fill::zeros // Zeros
    )
{
}

neural::layer::~layer()
{
}

neural::layer::width_t neural::layer::get_output_width() const
{
    return weights.n_rows;
}

neural::layer::width_t neural::layer::get_input_width() const
{
    return weights.n_cols;
}

arma::Col<float> neural::layer::evaluate(const arma::Col<float> &input)
{
    return
        (
            weights * input // Matrix multiplication, applying weights
            + biases // Adding biases
        ) // Compute the resultant column vector from passing through this layer
        .eval() // Force operation evaluation to complete. This is necessary for application of the activation function.
        .transform(SIGMOID_LAMBDA); // Applying the sigmoid function to each element
    ;
}

arma::Mat<float> &neural::layer::get_weights()
{
    return weights;
}

arma::Col<float> &neural::layer::get_biases()
{
    return biases;
}

const arma::Mat<float> &neural::layer::get_weights() const
{
    return weights;
}

const arma::Col<float> &neural::layer::get_biases() const
{
    return biases;
}

void neural::layer::set_weights(const arma::Mat<float> &_weights)
{
    this->weights = _weights;
}

void neural::layer::set_biases(const arma::Col<float> &_biases)
{
    this->biases = _biases;
}