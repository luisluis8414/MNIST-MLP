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

The network, backpropagation and training loop use no ML library. Trained on MNIST, the network gets 96.48 % of the 10,000 test images right.

The code is split into two projects: `mlp` is a shared library with the network itself (perceptrons, layers, backpropagation, saving and loading), and `train` is a console app that trains a model and evaluates it on the test set.

## Results

On the test set the 5 and the 9 are the hardest digits, the 0 and the 1 the easiest:

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
| all   | 96.48 %  | 9648/10000    |

## Network and training

The 784 inputs (28 × 28 pixels) feed two sigmoid hidden layers with 128 and 64 neurons, followed by a softmax output layer with 10 neurons.

The loss is cross-entropy, so the gradient of softmax plus cross-entropy is simply `output - target`. Weights and biases start uniformly in [-1, 1] and are updated by plain stochastic gradient descent with a learning rate of 0.01, one sample at a time. Training runs for at most 100 epochs and stops early once the validation accuracy has not improved by at least 0.1 percentage points for 5 epochs.

The data is [MNIST in CSV format](https://pjreddie.com/projects/mnist-in-csv/), 70,000 images in total. The first 80 % of `mnist_train.csv` (48,000 images) are used for training and the last 20 % (12,000) for validation. The 10,000 images in `mnist_test.csv` are only used for the final evaluation.

## Build

The project was developed and tested on Windows with Visual Studio 2022. It needs no third-party libraries, only [Premake5](https://premake.github.io/) in your `PATH`.

Run the build script from the project root in a Developer Command Prompt:

```batch
.\scripts\build.bat Release   :: default
.\scripts\build.bat Debug
.\scripts\build.bat All
```

Or build manually:

```batch
premake5 vs2022
msbuild MultiLayerPerception.sln /p:Configuration=Release /p:Platform=x64
```

Use Release for training; Debug is 5 to 10 times slower.

## Usage

Download `mnist_train.csv` and `mnist_test.csv` from [pjreddie.com](https://pjreddie.com/projects/mnist-in-csv/) into `data/`, and run everything from the repository root.

To train a new model:

```batch
.\bin\Release\train\train.exe
```

The model is saved to `models/` with the training parameters in its name, for example `models/model_0.01_100_60000_128_64`.

To evaluate the pre-trained model or another one:

```batch
.\bin\Release\train\train.exe eval            :: pre-trained model
.\bin\Release\train\train.exe eval <model>    :: other model
```

This prints the predictions for the first 20 test images, the overall accuracy and the accuracy per digit.
