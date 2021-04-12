# ESP-FACE

This is a component which provides API for **Neural Network** and some deep-learning **Applications**, such as Cat Face Detection, Human Face Detection and Human Face Recognition. It can be used as a component of some project as it doesn't support any interface of peripherals. By default, it works along with ESP-WHO, which is a project-level repository. 

> ESP-FACE is facing a thoroughly change for in coming ESP32-S3. This change has been proved performance improvement. However, it's still an alpha version. We are working hard on perfecting it. The document tells what has been done and what will be done. By now, it supports ESP32-S3-beta2/beta3 only. We hope you'll have good experience.
>
> Please use ESP-IDF release/v4.3 branch for ESP32-S3-beta2. Check [ESP-IDF](https://github.com/espressif/esp-idf) for ESP-IDF requirement of other chips.
>
> As this is a alpha version, not all functions are ready. Stable API has some limitation:
>
> - Conv2D:
>   - filter [3, 3, >=3, 8x] and [1, 1, >=3, 8x] with ReLU and Leaky_ReLU
>   - filter [1, 1, >=3, not 8x] with ReLU
> - DepthwiseConv2D:
>   - filter [3, 3, 8x, 1] with ReLU and Leaky_ReLU
> - Concat2D:
>   - no limitation



## Neural Network

ESP-FACE only supports quantization calculation. Element is quantized in following rule.

$$
element_{float} * 2^{exponent} = element_{quantized}
$$



| API                   | [ESP32](./lib/esp32) | [ESP32-S2](./lib/esp32s2) | [ESP32-C3](./lib/esp32c3) | [ESP32-S3-beta2/beta3](./lib/esp32s3) | [ESP32-S3](./lib/esp32s3) |
| --------------------- | :------------------: | :-----------------------: | :-----------------------: | :-----------------------------------: | :-----------------------: |
| Conv2D                |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| DepthwiseConv2D       |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| GlobalDepthwiseConv2D |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| Concat2D              |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| ReLU                  |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| LeakyReLU             |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |              **16-bit**               |     **16-bit**, 8-bit     |
| PReLU                 |    16-bit, 8-bit     |       16-bit, 8-bit       |       16-bit, 8-bit       |                16-bit                 |     **16-bit**, 8-bit     |


> In which, *16-bit* means 16-bit-quantization. *8-bit* means 8-bit-quantization. In bold means supported, on the contrary, means not supported yet but will be supported.

Some specific operations, e.g. Conv2D_3x3, Conv2D_1x1 and DepthwiseConv2D_3x3, are optimized and recommended strongly. Please click the chip name for more details.



## Build Your Own Model

[Here](./tutorial) is a tutorial to teach you how to build your own model with ESP-FACE step by step.



## Application

| Application            | API Navigation                                                                     | Example Navigation |
| ---------------------- | ---------------------------------------------------------------------------------- | ------------------ |
| Cat Face Detection     | [./include/model/cat_face_detector.hpp](./include/model/cat_face_detector.hpp)     |                    |
| Human Face Detection   | [./include/model/human_face_detector.hpp](./include/model/human_face_detector.hpp) |                    |

