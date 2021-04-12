# ESP32-S3 Special Optimization



## ESP32-S3-beta2/beta3

### Conv2D

- with filter [3, 3, >=3, 8x]
  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU
- with filter [1, 1, >=3, 8x]

  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU

- with filter [1, 1, >=3, not 8x]
  - [x] with/without bias
  - [x] with/without ReLU



### DepthwiseConv2D

- with filter [3, 3, 8x, 1]
  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU



## ESP32-S3

### Conv2D

- with filter [3, 3, >=3, 8x]
  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU, PReLU
- with filter [1, 1, >=3, 8x]

  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU, PReLU

- with filter [1, 1, >=3, >0]
  - [x] with/without bias
  - [x] with/without ReLU



### DepthwiseConv2D

- with filter [3, 3, 8x, 1]
  - [x] with/without bias
  - [x] with/without ReLU, LeakyReLU, PReLU