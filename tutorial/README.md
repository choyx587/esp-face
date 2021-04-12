# Tutorial

This tutorial will teaching you with an example to build your own model with ESP-FACE step by step. The example is a model for MNIST classification.



#### Step 1: Save Model Coefficient

Save the float point coefficients of model into npy files layer by layer, e.g.

```python
import numpy
numpy.save(file=f'{root}/{layer_name}_filter.npy', arr=filter) # filter must be numpy.ndarray
numpy.save(file=f'{root}/{layer_name}_bias.npy', arr=bias)     # bias must be numpy.ndarray
```

> NOTE: make sure the coefficients are in numpy.ndarray type.

**e.g.** `./model/npy/` contains the npy files of the example.



#### Step 2: Write Model Configuration

Write a config.json file for model configuration layer by layer, the format is 

```json
{
    "layer_name": {             // must be the same as the corresponding npy file
        "type": "conv2d",       // "conv2d", "depthwise_conv2d", "global_depthwise_conv2d" only by now
        "filter": -99,          // exponent of filter. If it equals to -99, the convert tool will select an exponent to project filter to whole quantization range.
        "element_width": 16,    // quantization element width
        "bias": -10,            // exponent of bias, which must be equal to output's
        "activation": {
            "type": "relu"      // "relu", "leaky_relu" only by now
            "exponent": -99
        }
    }, 
    ... ...
}
```

**e.g.** `./model/npy/config.json`

```json
{
    "l1": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -3, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l2_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l2_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -2
    }, 
    "l3_a_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l3_a_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -13
    }, 
    "l3_b_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l3_b_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -13
    }, 
    "l3_c_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l3_c_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -13
    }, 
    "l3_d_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l3_d_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -12
    }, 
    "l3_e_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "relu"
        }
    }, 
    "l3_e_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -13
    }, 
    "l4_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "leaky_relu", 
            "exponent": -99
        }
    }, 
    "l4_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -11
    }, 
    "l5_depth": {
        "type": "depthwise_conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "activation": {
            "type": "leaky_relu", 
            "exponent": -99
        }
    }, 
    "l5_compress": {
        "type": "conv2d", 
        "filter": -99, 
        "element_width": 16, 
        "bias": -10
    }
}
```



#### Step 3: Convert Model Coefficient

Make sure that coefficient npy files and configuration file are ready and in the same folder. Then, `convert.py` can help to convert coefficient into C++ code.

```python
# -i: where the coefficient and configuration files saved
# -n: generated filename
# -o: where the generated file saved
python convert.py -i input_root/ -n filename -o output_root/
```

**e.g.** 

```python
# under tutorial root
python ../convert.py -i ./model/npy/ -n mnist_coefficient -o ./model/
```

`mnist_coefficient.cpp` and `mnist_coefficient.hpp` can be found in `./model/`



#### Step 4: Build Model

Build a model by deriving Model class in `"dl_layer_model.hpp"`. Two pure abstract functions have to be implemented:

`void build(Feature<input_t> &input)`: for passing on the change of shape and padding layer by layer

`void call(Feature<input_t> &input)`: for running the model layer by layer

**e.g.** `./model/mnist_model.hpp`

```c++
#pragma once
#include "dl_layer_model.hpp"
#include "dl_layer_conv2d.hpp"
#include "dl_layer_depthwise_conv2d.hpp"
#include "dl_layer_concat2d.hpp"
#include "mnist_coefficient.hpp"
#include <stdint.h>

using namespace dl;
using namespace layer;
using namespace mnist_coefficient;

class MNIST : public Model<int16_t, int16_t>
{
private:
    Conv2D<int16_t, int16_t> l1;
    DepthwiseConv2D<int16_t, int16_t> l2_depth;
    Conv2D<int16_t, int16_t> l2_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_a_depth;
    Conv2D<int16_t, int16_t> l3_a_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_b_depth;
    Conv2D<int16_t, int16_t> l3_b_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_c_depth;
    Conv2D<int16_t, int16_t> l3_c_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_d_depth;
    Conv2D<int16_t, int16_t> l3_d_compress;
    DepthwiseConv2D<int16_t, int16_t> l3_e_depth;
    Conv2D<int16_t, int16_t> l3_e_compress;
    Concat2D<int16_t> l3_concat;
    DepthwiseConv2D<int16_t, int16_t> l4_depth;
    Conv2D<int16_t, int16_t> l4_compress;
    DepthwiseConv2D<int16_t, int16_t> l5_depth;

public:
    Conv2D<int16_t, int16_t> l5_compress;
    MNIST() : l1(Conv2D<int16_t, int16_t>(-3, get_l1_filter(), get_l1_bias(), get_l1_relu(), PADDING_VALID, 2, 2, "l1")),
              l2_depth(DepthwiseConv2D<int16_t, int16_t>(-1, get_l2_depth_filter(), NULL, get_l2_depth_relu(), PADDING_SAME, 2, 2, "l2_depth")),
              l2_compress(Conv2D<int16_t, int16_t>(-2, get_l2_compress_filter(), get_l2_compress_bias(), NULL, PADDING_SAME, 1, 1, "l2_compress")),
              l3_a_depth(DepthwiseConv2D<int16_t, int16_t>(-1, get_l3_a_depth_filter(), NULL, get_l3_a_depth_relu(), PADDING_VALID, 1, 1, "l3_a_depth")),
              l3_a_compress(Conv2D<int16_t, int16_t>(-13, get_l3_a_compress_filter(), get_l3_a_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_a_compress")),
              l3_b_depth(DepthwiseConv2D<int16_t, int16_t>(-1, get_l3_b_depth_filter(), NULL, get_l3_b_depth_relu(), PADDING_VALID, 1, 1, "l3_b_depth")),
              l3_b_compress(Conv2D<int16_t, int16_t>(-13, get_l3_b_compress_filter(), get_l3_b_compress_bias(), NULL, PADDING_VALID, 1, 1, "l3_b_compress")),
              l3_c_depth(DepthwiseConv2D<int16_t, int16_t>(-13, get_l3_c_depth_filter(), NULL, get_l3_c_depth_relu(), PADDING_SAME, 1, 1, "l3_c_depth")),
              l3_c_compress(Conv2D<int16_t, int16_t>(-13, get_l3_c_compress_filter(), get_l3_c_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_c_compress")),
              l3_d_depth(DepthwiseConv2D<int16_t, int16_t>(-12, get_l3_d_depth_filter(), NULL, get_l3_d_depth_relu(), PADDING_SAME, 1, 1, "l3_d_depth")),
              l3_d_compress(Conv2D<int16_t, int16_t>(-12, get_l3_d_compress_filter(), get_l3_d_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_d_compress")),
              l3_e_depth(DepthwiseConv2D<int16_t, int16_t>(-11, get_l3_e_depth_filter(), NULL, get_l3_e_depth_relu(), PADDING_SAME, 1, 1, "l3_e_depth")),
              l3_e_compress(Conv2D<int16_t, int16_t>(-13, get_l3_e_compress_filter(), get_l3_e_compress_bias(), NULL, PADDING_SAME, 1, 1, "l3_e_compress")),
              l3_concat("l3_concat"),
              l4_depth(DepthwiseConv2D<int16_t, int16_t>(-12, get_l4_depth_filter(), NULL, get_l4_depth_leaky_relu(), PADDING_VALID, 1, 1, "l4_depth")),
              l4_compress(Conv2D<int16_t, int16_t>(-11, get_l4_compress_filter(), get_l4_compress_bias(), NULL, PADDING_VALID, 1, 1, "l4_compress")),
              l5_depth(DepthwiseConv2D<int16_t, int16_t>(-11, get_l5_depth_filter(), NULL, get_l5_depth_leaky_relu(), PADDING_VALID, 1, 1, "l5_depth")),
              l5_compress(Conv2D<int16_t, int16_t>(-10, get_l5_compress_filter(), get_l5_compress_bias(), NULL, PADDING_VALID, 1, 1, "l5_compress")) {}

    void build(Feature<int16_t> &input)
    {
        this->l1.build(input);
        this->l2_depth.build(this->l1.output);
        this->l2_compress.build(this->l2_depth.output);
        this->l3_a_depth.build(this->l2_compress.output);
        this->l3_a_compress.build(this->l3_a_depth.output);
        this->l3_b_depth.build(this->l2_compress.output);
        this->l3_b_compress.build(this->l3_b_depth.output);
        this->l3_c_depth.build(this->l3_b_compress.output);
        this->l3_c_compress.build(this->l3_c_depth.output);
        this->l3_d_depth.build(this->l3_b_compress.output);
        this->l3_d_compress.build(this->l3_d_depth.output);
        this->l3_e_depth.build(this->l3_d_compress.output);
        this->l3_e_compress.build(this->l3_e_depth.output);
        this->l3_concat.build({&this->l3_a_compress.output, &this->l3_c_compress.output, &this->l3_e_compress.output});
        this->l4_depth.build(this->l3_concat.output);
        this->l4_compress.build(this->l4_depth.output);
        this->l5_depth.build(this->l4_compress.output);
        this->l5_compress.build(this->l5_depth.output);

        this->l3_concat.backward();
    }

    void call(Feature<int16_t> &input)
    {
        this->l1.call(input);
        input.free_element();
        this->l1.output.print2d(0, 10, 0, 10, 0, "l1_output");

        this->l2_depth.call(this->l1.output);
        this->l1.output.free_element();

        this->l2_compress.call(this->l2_depth.output);
        this->l2_depth.output.free_element();

        this->l3_a_depth.call(this->l2_compress.output);
        // this->l2_compress.output.free_element();

        this->l3_concat.calloc_element(); // calloc a memory for layers concat in future.

        this->l3_a_compress.call(this->l3_a_depth.output);
        this->l3_a_depth.output.free_element();

        this->l3_b_depth.call(this->l2_compress.output);
        this->l2_compress.output.free_element();

        this->l3_b_compress.call(this->l3_b_depth.output);
        this->l3_b_depth.output.free_element();

        this->l3_c_depth.call(this->l3_b_compress.output);
        // this->l3_b_compress.output.free_element();

        this->l3_c_compress.call(this->l3_c_depth.output);
        this->l3_c_depth.output.free_element();

        this->l3_d_depth.call(this->l3_b_compress.output);
        this->l3_b_compress.output.free_element();

        this->l3_d_compress.call(this->l3_d_depth.output);
        this->l3_d_depth.output.free_element();

        this->l3_e_depth.call(this->l3_d_compress.output);
        this->l3_d_compress.output.free_element();

        this->l3_e_compress.call(this->l3_e_depth.output);
        this->l3_e_depth.output.free_element();

        this->l4_depth.call(this->l3_concat.output);
        this->l3_concat.output.free_element();

        this->l4_compress.call(this->l4_depth.output);
        this->l4_depth.output.free_element();

        this->l5_depth.call(this->l4_compress.output);
        this->l4_compress.output.free_element();

        this->l5_compress.call(this->l5_depth.output);
        this->l5_depth.output.free_element();
    }
};
```



#### Step 5: Run Model

Create a model object and run its forward()

**e.g.** `./main/main.cpp`

```c++
#include <stdio.h>
#include <stdlib.h>

#include "esp_system.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include "mnist_model.hpp"

__attribute__((aligned(16))) int16_t example_element[] = {...};

extern "C" void app_main(void)
{
    // input
    Feature<int16_t> input;
    input.set_element((int16_t *)example_element).set_exponent(0).set_shape({28, 28, 3}).set_auto_free(false);

    // model forward
    MNIST model;
    model.forward(input);

    // parse
    int16_t *score = model.l5_compress.output.get_element_ptr();
    int16_t max_score = score[0];
    int max_index = 0;
    printf("%d, ", max_score);
    for (size_t i = 1; i < 10; i++)
    {
        printf("%d, ", score[i]);
        if (score[i] > max_score)
        {
            max_score = score[i];
            max_index = i;
        }
    }
    printf("\nPrediction Result: %d\n", max_index);
}
```

flash and monitor

```bash
$ idf.py -p /dev/ttyUSB0 flash monitor # NOTICE: please select a right device

-7166, -9783, -12293, -11405, -12351, -1363, -11715, -116, -11436, 7851, 
Prediction Result: 9
```

