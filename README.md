<div align="center">

<h1>MNIST_NN</h1>

<p><b>A multi-layer perceptron written from scratch in C++ that recognises handwritten digits.</b></p>

<p>
<img src="https://img.shields.io/badge/C++-17-00599C?logo=cplusplus&logoColor=white" alt="C++17">
<img src="https://img.shields.io/badge/NN-from%20scratch-3B6EA8" alt="NN from scratch">
<img src="https://img.shields.io/badge/MNIST-96.48%25%20test%20accuracy-2E8B57" alt="96.48% test accuracy">
</p>

<img src="assets/demo.gif" alt="Drawing a digit and letting the network recognise it" width="600">

</div>

# Overview

The network, backpropagation and training loop are implemented without any ML library. It is trained on the MNIST dataset and reaches **96.48 %** accuracy on the 10,000 test images.

The repository contains two projects:

- **mlp**: Shared library with the neural network (perceptrons, layers, backpropagation, save/load)
- **train**: Console app to train a model and evaluate it on the test set

# Results

| Digit | Accuracy | Correct/Total |
|:-----:|:--------:|:-------------:|
| 0     | 98.37 %  | 964/980       |
| 1     | 98.59 %  | 1119/1135     |
| 2     | 95.54 %  | 986/1032      |
| 3     | 96.34 %  | 973/1010      |
| 4     | 96.44 %  | 947/982       |
| 5     | 94.73 %  | 845/892       |
| 6     | 96.35 %  | 923/958       |
| 7     | 96.69 %  | 994/1028      |
| 8     | 96.10 %  | 936/974       |
| 9     | 95.24 %  | 961/1009      |
| **All** | **96.48 %** | **9648/10000** |

# Network

| Layer          | Neurons | Activation |
|:---------------|:-------:|:----------:|
| Input          | 784 (28 × 28 pixels) | – |
| Hidden 1       | 128     | Sigmoid    |
| Hidden 2       | 64      | Sigmoid    |
| Output         | 10      | Softmax    |

## Training

- **Loss**: Cross-entropy (gradient of softmax + cross-entropy is `output - target`)
- **Optimiser**: Stochastic gradient descent, learning rate 0.01, one sample at a time
- **Initialisation**: Weights and biases uniform in [-1, 1]
- **Early stopping**: Stops when the validation accuracy has not improved by at least 0.1 percentage points for 5 epochs (max. 100 epochs)

## Dataset

[MNIST in CSV format](https://pjreddie.com/projects/mnist-in-csv/), 70,000 images:

- **Training**: 48,000 images (first 80 % of `mnist_train.csv`)
- **Validation**: 12,000 images (last 20 % of `mnist_train.csv`)
- **Test**: 10,000 images (`mnist_test.csv`, only used for the final evaluation)

# Project Structure

```
MNIST_NN/
├── mlp/               - Neural network library (mlp.dll)
│   ├── include/       - mlp.h, perceptron.h
│   └── src/
├── train/             - Training and evaluation app
│   ├── main.cpp       - train / eval commands
│   ├── dataset.cpp    - MNIST CSV loader
│   └── config.hpp     - Network configuration
├── models/            - Pre-trained model
├── data/              - Place the MNIST CSV files here (not in git)
├── assets/            - Demo GIF
├── scripts/           - Build script
└── premake5.lua
```

# Build

Developed and tested on Windows with Visual Studio 2022.

## Dependencies

No third-party libraries are needed, only [Premake5](https://premake.github.io/) in your `PATH`.

## Build Script

Run from the project root in a Developer Command Prompt:

```batch
.\scripts\build.bat Release   :: default
.\scripts\build.bat Debug
.\scripts\build.bat All
```

Or manually:

```batch
premake5 vs2022
msbuild MultiLayerPerception.sln /p:Configuration=Release /p:Platform=x64
```

Use **Release** for training, Debug is 5–10× slower.

# Usage

Download `mnist_train.csv` and `mnist_test.csv` from [here](https://pjreddie.com/projects/mnist-in-csv/) to `data/`. Run everything from the repository root.

## Train a New Model

```batch
.\bin\Release\train\train.exe
```

The model is saved to `models/` with the training parameters in its name, e.g. `models/model_0.01_100_60000_128_64`.

## Evaluate a Model

```batch
.\bin\Release\train\train.exe eval            :: pre-trained model
.\bin\Release\train\train.exe eval <model>    :: other model
```

Prints the predictions for the first 20 test images, the overall accuracy and the accuracy per digit.

# Future Improvements

- Convolutional layers for better spatial features
- ReLU / Leaky ReLU activations
- Mini-batch training
- Regularisation: dropout, batch normalisation, L1/L2
