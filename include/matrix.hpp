#pragma once

#include <array>
#include <thread>
#include <numeric>

namespace xneur
{
    template <size_t N>
    class colvec : public std::array<float, N>
    {
    public:
        const row_count = N;
        const col_count = 1;
    public:
        colvec<N> operator+(const colvec<N> &v) const
        {
            colvec<N> result;
            std::transform(this->begin(), this->end(), v.begin(), result.begin(), std::plus<float>());
            return result;
        }
    };

    template <size_t N>
    class rowvec : public std::array<float, N>
    {
    public:
        const row_count = 1;
        const col_count = N;
    public:
        float operator*(const colvec<N> &v) const
        {
            return std::inner_product(this->begin(), this->end(), v.begin(), float(0));
        }
    };

    template <size_t M, size_t N>
    class matrix : public std::array<rowvec<N>, M>
    {
    public:
        static const row_count = M;
        static const col_count = N;
    public:
        colvec<M> operator*(const colvec<N> &v) const
        {
            colvec<M> result;
            std::transform(this->begin(), this->end(), result.begin(), [&v](const rowvec<N>& row) {
                return row * v;
            });
            return result;
        }

        /// @brief Iterator generator for matrix cells (not rows)
        /// @return Iterator to the first cell
        auto cell_begin() { return this->front().begin(); } 
        /// @brief Iterator generator for matrix cells (not rows)
        /// @return Iterator to the end cell
        auto cell_end() { return this->back().end(); }
        /// @brief Iterator generator for matrix cells (not rows)
        /// @return Iterator to the first cell (const version)
        auto cell_begin() const { return this->front().begin(); }
        /// @brief Iterator generator for matrix cells (not rows)
        /// @return Iterator to the end cell (const version)
        auto cell_end() const { return this->back().end(); }
    };
} // namespace xneur