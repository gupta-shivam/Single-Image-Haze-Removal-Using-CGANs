# Single Image Haze Removal using CGANs
A fully end-to-end neural network model based on CGANs(Conditional Generative Adversarial Networks) using tensorflow.

## Prerequisites

* TensorFlow 1.4.1 or later
* Python 3

## Demo

* Test the model

```sh
 sh demo.sh data/indoor results/indoor models/Hazy2GT.pb
```


*  You can use this model for your own images. 

```sh
sh demo.sh input_folder output_folder model_name
```
