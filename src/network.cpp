#include "network.hpp"
#include "cost.hpp"

#include <vector>
#include <thread>

neural::network::network() :
    depth(0),
    layers(nullptr)
{
}

neural::network::network(const depth_t &depth, const layer::width_t *widths) :
    depth(depth),
    layers(new layer[depth])
{
    for (depth_t i = 1; i < depth; i++)
    {
        layers[i] = layer(widths[i-1], widths[i]);
    }
}

neural::network::~network()
{
    delete[] layers;
}

neural::network::gradient_t neural::network::backpropagate(const training_data_set_t &training_data)
{
    const auto cost_function = COST_FUNCTION_LAMBDA;
    const std::vector<training_data_t>& data_points = training_data.data_points;

    gradient_t total_gradient(this->depth);

    for (const training_data_t& data_point : data_points)
    {
        gradient_t local_gradient(this->depth);
        const arma::Col<float> network_output = forward_propagate(data_point.input);
        const float cost = cost_function(data_point.expected_output.begin(), network_output.begin(), this->layers[this->depth].get_output_width());
    }

    total_gradient /= data_points.size();
    return total_gradient;
}

neural::layer::width_t neural::network::get_input_width() const
{
    return layers[0].get_input_width();
}

neural::layer::width_t neural::network::get_output_width() const
{
    return layers[this->depth].get_output_width();
}

arma::Col<float> neural::network::forward_propagate(data_col_t input)
{
    arma::Col<float> output = input;
    for (depth_t i = 0; i < this->depth; i++)
    {
        output = layers[i].evaluate(output);
    }
    return output;
}

float neural::network::cost(training_data_t training_data)
{
    const auto cost_function = COST_FUNCTION_LAMBDA;

    arma::Col<float> network_output = forward_propagate(training_data.input);

    const float* network_output_pointer = training_data.input.begin();
    const float* expected_output_pointer = training_data.expected_output.begin();

    cost_function(expected_output_pointer, network_output_pointer, this->layers[this->depth].get_output_width());
}

float neural::network::cost(const training_data_set_t &training_data)
{
    const std::vector<training_data_t>& data_points = training_data.data_points;

    std::vector<std::thread> threads;
    for (const training_data_t& data_point : data_points)
    {
        threads.push_back(std::thread([this, &data_point]() {
            this->cost(data_point);
        }));
    }

    for (auto& thread : threads)
    {
        thread.join();
    }
}

void neural::network::train(const training_data_set_t &training_data, const float &learning_rate)
{
}

neural::network::gradient_t::gradient_t(const depth_t &depth) :
    d_weights(new arma::Mat<float>[depth]),
    d_biases(new arma::Col<float>[depth]),
    depth(depth)
{
}

neural::network::gradient_t::~gradient_t()
{
    delete[] d_weights;
    delete[] d_biases;
}

neural::network::gradient_t &neural::network::gradient_t::operator+=(const gradient_t &other)
{
    for (depth_t i = 0; i < this->depth; i++)
    {
        this->d_weights[i] += other.d_weights[i];
        this->d_biases[i] += other.d_biases[i];
    }
}

neural::network::gradient_t &neural::network::gradient_t::operator/=(const float &divisor)
{
    for (depth_t i = 0; i < this->depth; i++)
    {
        this->d_weights[i] /= divisor;
        this->d_biases[i] /= divisor;
    }
}
