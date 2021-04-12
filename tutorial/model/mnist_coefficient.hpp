#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace mnist_coefficient
{
	const dl::Filter<int16_t> *get_l1_filter();
	const dl::Bias<int16_t> *get_l1_bias();
	const dl::ReLU<int16_t> *get_l1_relu();
	const dl::Filter<int16_t> *get_l2_depth_filter();
	const dl::ReLU<int16_t> *get_l2_depth_relu();
	const dl::Filter<int16_t> *get_l2_compress_filter();
	const dl::Bias<int16_t> *get_l2_compress_bias();
	const dl::Filter<int16_t> *get_l3_a_depth_filter();
	const dl::ReLU<int16_t> *get_l3_a_depth_relu();
	const dl::Filter<int16_t> *get_l3_a_compress_filter();
	const dl::Bias<int16_t> *get_l3_a_compress_bias();
	const dl::Filter<int16_t> *get_l3_b_depth_filter();
	const dl::ReLU<int16_t> *get_l3_b_depth_relu();
	const dl::Filter<int16_t> *get_l3_b_compress_filter();
	const dl::Bias<int16_t> *get_l3_b_compress_bias();
	const dl::Filter<int16_t> *get_l3_c_depth_filter();
	const dl::ReLU<int16_t> *get_l3_c_depth_relu();
	const dl::Filter<int16_t> *get_l3_c_compress_filter();
	const dl::Bias<int16_t> *get_l3_c_compress_bias();
	const dl::Filter<int16_t> *get_l3_d_depth_filter();
	const dl::ReLU<int16_t> *get_l3_d_depth_relu();
	const dl::Filter<int16_t> *get_l3_d_compress_filter();
	const dl::Bias<int16_t> *get_l3_d_compress_bias();
	const dl::Filter<int16_t> *get_l3_e_depth_filter();
	const dl::ReLU<int16_t> *get_l3_e_depth_relu();
	const dl::Filter<int16_t> *get_l3_e_compress_filter();
	const dl::Bias<int16_t> *get_l3_e_compress_bias();
	const dl::Filter<int16_t> *get_l4_depth_filter();
	const dl::ReLU<int16_t> *get_l4_depth_leaky_relu();
	const dl::Filter<int16_t> *get_l4_compress_filter();
	const dl::Bias<int16_t> *get_l4_compress_bias();
	const dl::Filter<int16_t> *get_l5_depth_filter();
	const dl::ReLU<int16_t> *get_l5_depth_leaky_relu();
	const dl::Filter<int16_t> *get_l5_compress_filter();
	const dl::Bias<int16_t> *get_l5_compress_bias();
}
