#pragma once

#include <vector>
#include "dl_constant.hpp"
#include "dl_variable.hpp"
#include "dl_tool.hpp"
#include "dl_nn_depthwise_conv2d.hpp"

namespace dl
{
    namespace layer
    {
        /**
         * @brief ReLU(DepthwiseConv2D(filter, input) + bias)
         * 
         * @tparam input_t      type of input
         * @tparam output_t     type of output
         */
        template <typename input_t, typename output_t>
        class DepthwiseConv2D : public Layer
        {
        private:
            const Filter<output_t> *filter;    /*<! filter >*/
            const int stride_y;                /*<! stride in height >*/
            const int stride_x;                /*<! stride in width >*/
            const padding_type_t padding_type; /*<! padding type >*/
            const Bias<output_t> *bias;        /*<! bias >*/
            const ReLU<output_t> *relu;        /*<! relu >*/
            std::vector<int> padding;          /*<! padding >*/

        public:
            Feature<output_t> output; /*<! output >*/

            /**
             * @brief Construct a new Depthwise Conv2D object
             * 
             * @param output_exponent 
             * @param filter 
             * @param bias 
             * @param relu 
             * @param padding_type 
             * @param stride_y 
             * @param stride_x 
             * @param name 
             */
            DepthwiseConv2D(const int output_exponent,
                            const Filter<output_t> *filter,
                            const Bias<output_t> *bias = NULL,
                            const ReLU<output_t> *relu = NULL,
                            const padding_type_t padding_type = PADDING_VALID,
                            const int stride_y = 1,
                            const int stride_x = 1,
                            const char *name = NULL) : Layer(name),
                                                       filter(filter),
                                                       stride_y(stride_y),
                                                       stride_x(stride_x),
                                                       padding_type(padding_type),
                                                       bias(bias),
                                                       relu(relu)
            {
                this->output.set_exponent(output_exponent);
            }

            /**
             * @brief Destroy the Depthwise Conv2D object
             * 
             */
            ~DepthwiseConv2D() {}

            /**
             * @brief update output shape and padding
             * 
             * @param input 
             */
            void build(Feature<input_t> &input)
            {
                this->build(this->output, input);
            }

            /**
             * @brief update output shape and padding
             * 
             * @param output 
             * @param input 
             */
            void build(Feature<output_t> &output, Feature<input_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                std::vector<int> output_shape = tool2d::get_output_shape(input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type, true);
                output.set_shape(output_shape);

                this->padding = tool2d::get_pad_size(output_shape, input.shape, this->filter->shape_with_dilation, this->stride_y, this->stride_x, this->padding_type);
                input.set_padding(this->padding);
            }

            /**
             * @brief calloc output's element if not calloc yet. Call depthwise_conv2d
             * 
             * @param input 
             * @return Feature<output_t>& 
             */
            Feature<output_t> &call(Feature<input_t> &input)
            {
#if CONFIG_DEBUG_MODE
                printf("%s:\n", this->name);
                dl::tool::Latency latency;
                latency.start();
#endif
                this->output.calloc_element();
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print("\tcalloc");
#endif
                this->call(this->output, input);
                return this->output;
            }

            /**
             * @brief Call depthwise_conv2d
             * 
             * @param input 
             * @return Feature<output_t>& 
             */
            void call(Feature<output_t> &output, Feature<input_t> &input)
            {
#if CONFIG_DEBUG_MODE
                dl::tool::Latency latency;
                latency.start();
#endif
                if (this->bias)
                {
                    if (this->relu)
                        nn::depthwise_conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x, *(this->bias), *(this->relu));
                    else
                        nn::depthwise_conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x, *(this->bias));
                }
                else
                {
                    if (this->relu)
                        nn::depthwise_conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x, *(this->relu));
                    else
                        nn::depthwise_conv2d(output, input, this->padding, *(this->filter), this->stride_y, this->stride_x);
                }
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print(this->name);
#endif
            }
        };

        template <typename input_t, typename output_t>
        class GlobalDepthwiseConv2D : public Layer
        {
        private:
            const Filter<output_t> *filter;
            const Bias<output_t> *bias;
            const ReLU<output_t> *relu;

        public:
            Feature<output_t> output;

            GlobalDepthwiseConv2D(const int output_exponent,
                                  const Filter<output_t> *filter,
                                  const Bias<output_t> *bias = NULL,
                                  const ReLU<output_t> *relu = NULL,
                                  const char *name = NULL) : Layer(name),
                                                             filter(filter),
                                                             bias(bias),
                                                             relu(relu)
            {
                this->output.set_exponent(output_exponent);
            }
            ~GlobalDepthwiseConv2D() {}

            void build(Feature<input_t> &input)
            {
                this->build(this->output, input);
            }

            void build(Feature<output_t> &output, Feature<input_t> &input)
            {
                assert(input.shape[0] > 0);
                assert(input.shape[1] > 0);

                std::vector<int> output_shape = tool2d::get_output_shape(input.shape, this->filter->shape_with_dilation, 1, 1, PADDING_VALID, true);
                output.set_shape(output_shape);
            }

            Feature<output_t> &call(Feature<input_t> &input, const char *message = NULL)
            {
#if CONFIG_DEBUG_MODE
                printf("%s:\n", this->name);
                dl::tool::Latency latency;
                latency.start();
#endif
                this->output.calloc_element();
                this->call(this->output, input);
#if CONFIG_DEBUG_MODE
                latency.end();
                latency.print(message);
#endif
                return this->output;
            }

            void call(Feature<output_t> &output, Feature<input_t> &input)
            {
                if (this->bias)
                {
                    nn::global_depthwise_conv2d(output, input, *(this->filter), *(this->bias));
                }
                else
                {
                    nn::global_depthwise_conv2d(output, input, *(this->filter));
                }
            }
        };
    } // namespace layer
} // namespace dl
