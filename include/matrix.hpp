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
    
        colvec<N> operator-(const colvec<N> &v) const
        {
            colvec<N> result;
            std::transform(this->begin(), this->end(), v.begin(), result.begin(), std::minus<float>());
            return result;
        }

        colvec<N> operator*(float scalar) const
        {
            colvec<N> result;
            std::transform(this->begin(), this->end(), result.begin(), [scalar](float x) { return x * scalar; });
            return result;
        }
    public:
        colvec<N> &operator+=(const colvec<N> &v)
        {
            std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::plus<float>());
            return *this;
        }

        colvec<N> &operator-=(const colvec<N> &v)
        {
            std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::minus<float>());
            return *this;
        }

        colvec<N> &operator*=(float scalar)
        {
            std::transform(this->begin(), this->end(), this->begin(), [scalar](float x) { return x * scalar; });
            return *this;
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
    
        rowvec<N> operator+(const rowvec<N> &v) const
        {
            rowvec<N> result;
            std::transform(this->begin(), this->end(), v.begin(), result.begin(), std::plus<float>());
            return result;
        }

        rowvec<N> operator-(const rowvec<N> &v) const
        {
            rowvec<N> result;
            std::transform(this->begin(), this->end(), v.begin(), result.begin(), std::minus<float>());
            return result;
        }

        rowvec<N> operator*(float scalar) const
        {
            rowvec<N> result;
            std::transform(this->begin(), this->end(), result.begin(), [scalar](float x) { return x * scalar; });
            return result;
        }

    public:

        rowvec<N> &operator+=(const rowvec<N> &v)
        {
            std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::plus<float>());
            return *this;
        }

        rowvec<N> &operator-=(const rowvec<N> &v)
        {
            std::transform(this->begin(), this->end(), v.begin(), this->begin(), std::minus<float>());
            return *this;
        }

        rowvec<N> &operator*=(float scalar)
        {
            std::transform(this->begin(), this->end(), this->begin(), [scalar](float x) { return x * scalar; });
            return *this;
        }
    };

    template <size_t M, size_t N>
    class matrix : public std::array<rowvec<N>, M>
    {
    public:
        static const row_count = M;
        static const col_count = N;
    public: // Linear algebra

        /// @brief Transpose the matrix
        /// @return The transposed matrix
        matrix<N, M> transpose() const
        {
            matrix<N, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    result[j][i] = this->at(i).at(j);
                }
            }
            return result;
        }

    public: // Multiplication
        /// @brief Matrix-vector multiplication
        /// @param v The vector to multiply
        /// @return The result of the multiplication
        colvec<M> dot(const colvec<N> &v) const
        {
            colvec<M> result;
            for (size_t i = 0; i < M; i++)
            {
                result[i] = this->at(i) * v;
            }
            return result;
        }

        /// @brief Transposed matrix-vector multiplication
        /// @param v The vector to multiply
        /// @return The result of the multiplication
        colvec<N> transpose_dot(const colvec<M> &v) const
        {
            colvec<N> result;
            for (size_t i = 0; i < N; i++)
            {
                result[i] = 0;
                for (size_t j = 0; j < M; j++)
                {
                    result[i] += this->at(j).at(i) * v[j];
                }
            }
            return result;
        }
        /// @brief Transposed matrix-vector multiplication
        /// @param v The vector to multiply
        /// @return The result of the multiplication
        colvec<N> dot(const colvec<M> &v) const { return transpose_dot(v); }

        /// @brief Matrix-matrix multiplication
        /// @tparam P The number of columns in the second matrix
        /// @param m The matrix to multiply
        /// @return The result of the multiplication
        template<size_t P>
        matrix<M, P> dot(const matrix<N, P> &m) const
        {
            matrix<M, P> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < P; j++)
                {
                    result[i][j] = this->at(i).dot(m.transpose().at(j));
                }
            }
            return result;
        }

        /// @brief Transposed matrix-matrix multiplication
        /// @tparam P The number of columns in the second matrix
        /// @param m The matrix to multiply
        /// @return The result of the multiplication
        template<size_t P>
        matrix<N, P> transpose_dot(const matrix<M, P> &m) const
        {
            matrix<N, P> result;
            for (size_t i = 0; i < N; i++)
            {
                for (size_t j = 0; j < P; j++)
                {
                    result[i][j] = 0;
                    for (size_t k = 0; k < M; k++)
                    {
                        result[i][j] += this->at(k).at(i) * m[k][j];
                    }
                }
            }
            return result;
        }
        /// @brief Transposed matrix-matrix multiplication
        /// @tparam P The number of columns in the second matrix
        /// @param m The matrix to multiply
        /// @return The result of the multiplication
        template<size_t P>
        matrix<N, P> dot(const matrix<M, P> &m) const { return transpose_dot(m); }

        /// @brief Scalar multiplication
        /// @param scalar The scalar to multiply
        /// @return The result of the multiplication
        matrix<M, N> scalar_mult(float scalar) const
        {
            matrix<M, N> result = *this;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    result[i][j] *= scalar;
                }
            }
            return result;
        }
    public: // Addition
        matrix<M, N> add(const matrix<M, N> &m) const
        {
            matrix<M, N> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    result[i][j] = this->at(i).at(j) + m[i][j];
                }
            }
            return result;
        }

        matrix<M, N> subtract(const matrix<M, N> &m) const
        {
            matrix<M, N> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    result[i][j] = this->at(i).at(j) - m[i][j];
                }
            }
            return result;
        }
    public: // Operators
        colvec<M> operator*(const colvec<N> &v) const { return dot(v); }
        colvec<N> operator*(const colvec<M> &v) const { return transpose_dot(v); }

        template<size_t P>
        matrix<M, P> operator*(const matrix<N, P> &m) const { return dot(m); }
        template<size_t P>
        matrix<N, P> operator*(const matrix<M, P> &m) const { return transpose_dot(m); }

        matrix<M, N> operator+(const matrix<M, N> &m) const { return add(m); }
        matrix<M, N> operator-(const matrix<M, N> &m) const { return subtract(m); }

        matrix<M, N> operator*(float scalar) const { return scalar_mult(scalar); }
    public: // Modifying Operators
        matrix<M, N> &operator+=(const matrix<M, N> &m)
        {
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    this->at(i).at(j) += m[i][j];
                }
            }
            return *this;
        }

        matrix<M, N> &operator-=(const matrix<M, N> &m)
        {
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < N; j++)
                {
                    this->at(i).at(j) -= m[i][j];
                }
            }
            return *this;
        }
    
        matrix<M, N> &operator*=(float scalar)
        {
            *this = scalar_mult(scalar);
            return *this;
        }
    public: // Iterators
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

    template <size_t M>
    class matrix<M, M> : public std::array<rowvec<M>, M>
    {
    public:
        static const row_count = M;
        static const col_count = M;
    public: // Linear algebra

        matrix<M, M> transpose() const
        {
            matrix<M, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[j][i] = this->at(i).at(j);
                }
            }
            return result;
        }

    public: // Multiplication

        colvec<M> dot(const colvec<M> &v) const
        {
            colvec<M> result;
            for (size_t i = 0; i < M; i++)
            {
                result[i] = this->at(i) * v;
            }
            return result;
        }

        colvec<M> transpose_dot(const colvec<M> &v) const
        {
            colvec<M> result;
            for (size_t i = 0; i < M; i++)
            {
                result[i] = 0;
                for (size_t j = 0; j < M; j++)
                {
                    result[i] += this->at(j).at(i) * v[j];
                }
            }
            return result;
        }

        matrix<M, M> dot(const matrix<M, M> &m) const
        {
            matrix<M, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[i][j] = this->at(i).dot(m.transpose().at(j));
                }
            }
            return result;
        }

        matrix<M, M> transpose_dot(const matrix<M, M> &m) const
        {
            matrix<M, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[i][j] = 0;
                    for (size_t k = 0; k < M; k++)
                    {
                        result[i][j] += this->at(k).at(i) * m[k][j];
                    }
                }
            }
            return result;
        }

        matrix<M, M> scalar_mult(float scalar) const
        {
            matrix<M, M> result = *this;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[i][j] *= scalar;
                }
            }
            return result;
        }
    public: // Addition

        matrix<M, M> add(const matrix<M, M> &m) const
        {
            matrix<M, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[i][j] = this->at(i).at(j) + m[i][j];
                }
            }
            return result;
        }

        matrix<M, M> subtract(const matrix<M, M> &m) const
        {
            matrix<M, M> result;
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    result[i][j] = this->at(i).at(j) - m[i][j];
                }
            }
            return result;
        }

    public: // Operators

        colvec<M> operator*(const colvec<M> &v) const { return dot(v); }

        matrix<M, M> operator*(const matrix<M, M> &m) const { return dot(m); }

        matrix<M, M> operator*(float scalar) const { return scalar_mult(scalar); }

        matrix<M, M> operator+(const matrix<M, M> &m) const { return add(m); }

        matrix<M, M> operator-(const matrix<M, M> &m) const { return subtract(m); }

    public: // Modifying Operators

        matrix<M, M> &operator+=(const matrix<M, M> &m)
        {
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    this->at(i).at(j) += m[i][j];
                }
            }
            return *this;
        }

        matrix<M, M> &operator-=(const matrix<M, M> &m)
        {
            for (size_t i = 0; i < M; i++)
            {
                for (size_t j = 0; j < M; j++)
                {
                    this->at(i).at(j) -= m[i][j];
                }
            }
            return *this;
        }

        matrix<M, M> &operator*=(const matrix<M, M> &m)
        {
            *this = dot(m);
            return *this;
        }
    
        matrix<M, M> &operator*=(float scalar)
        {
            *this = scalar_mult(scalar);
            return *this;
        }
    public: // Iterators

        auto cell_begin() { return this->front().begin(); }

        auto cell_end() { return this->back().end(); }

        auto cell_begin() const { return this->front().begin(); }

        auto cell_end() const { return this->back().end(); }
    };    
} // namespace xneur