#include "network.hpp"
#include "cost.hpp"

#include <vector>
#include <thread>

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

    cost_function(expected_output_pointer, network_output_pointer, this->layers[this->depth].get_output_width())
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
