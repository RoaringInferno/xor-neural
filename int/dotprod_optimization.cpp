#include <iostream>
#include <vector>
#include <chrono>

#include <thread>
#include <numeric>
#include <functional>

const int num_trials_thread_benchmark = 100000;
const int num_trials_dotprod_benchmark = 10000;
const unsigned long start_dimension = 100;
const unsigned long dimension_increment = 100;


double benchmark_threads();
unsigned long marginally_benchmark_dotprod(double thread_cost);

int main()
{
    double nanoseconds_per_thread_generation = benchmark_threads();
    std::cout << "thread_cost = " << nanoseconds_per_thread_generation << " ns" << std::endl;
    unsigned long margin = marginally_benchmark_dotprod(nanoseconds_per_thread_generation);
    std::cout << "At dimension " << margin << ", dot_cost * (dim - 1) > thread_cost * dim" << std::endl;
}

double benchmark_threads() {
    std::vector<double> times(num_trials_thread_benchmark);

    for (int i = 0; i < num_trials_thread_benchmark; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        std::thread([]{}).join();
        auto end = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::nano>(end - start).count();
    }

    double average_time = std::accumulate(times.begin(), times.end(), 0.0) / num_trials_thread_benchmark;
    return average_time;
}

unsigned long marginally_benchmark_dotprod(double thread_cost) {
    unsigned long dimension = start_dimension;

    while (true) {
        std::vector<double> vec1(dimension, 1.0);
        std::vector<double> vec2(dimension, 1.0);
        std::vector<double> times(num_trials_dotprod_benchmark);

        for (int i = 0; i < num_trials_dotprod_benchmark; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            double dot_product = std::inner_product(vec1.begin(), vec1.end(), vec2.begin(), 0.0);
            auto end = std::chrono::high_resolution_clock::now();
            times[i] = std::chrono::duration<double, std::nano>(end - start).count();
        }

        double average_dotprod_time = std::accumulate(times.begin(), times.end(), 0.0) / num_trials_dotprod_benchmark;
        if (average_dotprod_time * (dimension - 1) > thread_cost * dimension) {
            return dimension;
        }

        dimension += dimension_increment;
    }
}