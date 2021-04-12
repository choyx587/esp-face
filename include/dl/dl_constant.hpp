#pragma once

#include "dl_define.hpp"
#include <vector>

namespace dl
{
    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Constant
    {
    public:
        const T *element;             /*<! The element of element */
        const int exponent;           /*<! The exponent of element */
        const std::vector<int> shape; /*<! The shape of element */

        Constant(const T *element, const int exponent, const std::vector<int> shape);

        /**
         * @brief Take numpy for example, this function print Constant[y_start:y_end, x_start:x_end, c]
         * 
         * @param y_start 
         * @param y_end 
         * @param x_start 
         * @param x_end 
         * @param c
         */
        void print2d(const int y_start, const int y_end, const int x_start, const int x_end, const int c, const char *message) const;
    };

    /**
     * @brief 
     * NOTE: The shape format of filter is fixed, but the element sequence depands on optimization.
     * For 1D: reserved
     * For 2D: shape format is [filter_height, filter_width, input_channel, output_channel]. dilation format is [height, width]
     *  
     * @tparam T 
     */
    template <typename T>
    class Filter : public Constant<T>
    {
    public:
        const std::vector<int> dilation;
        std::vector<int> shape_with_dilation;
        Filter(const T *element, const int exponent, const std::vector<int> shape, const std::vector<int> dilation = {1, 1});
    };

    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class Bias : public Constant<T>
    {
    public:
        using Constant<T>::Constant;
    };

    /**
     * @brief 
     * 
     * @tparam T 
     */
    template <typename T>
    class ReLU : public Constant<T>
    {
    public:
        const relu_type_t type; /*<! The type of ReLU */

        /**
         * @brief Construct a new ReLU object
         * 
         * @param type      ReLU type including RELU, LEAKY_RELU, PRELU
         * @param element   element of ReLU
         * @param exponent  exponent of element
         * @param shape     shape of element
         */
        ReLU(const relu_type_t type, const T *element = NULL, const int exponent = 0, const std::vector<int> shape = {0});
    };

} // namespace dl