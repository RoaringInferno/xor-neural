#include "network.hpp"

int main()
{
    typedef xneur::network<2, 1, 2, 1> network;

    network net;
    network::input_vector input = {0.05f, 0.1f};
    network::output_vector expected = {0.01f, 0.99f};
    net.train(input, expected);
    return 0;
}