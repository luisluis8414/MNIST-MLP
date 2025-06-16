/* ========================================================================== */
/* main.cpp                                                                   */
/* ========================================================================== */
/**
 * @file      main.cpp
 * @author    wehrberger
 * @date      16.06.2025
 *
 * @brief     Training and evaluation of the MLP on MNIST.
 *
 * Usage (run from the repository root):
 *   - `train`                 – train a new model and save it to models/
 *   - `train eval [model]`    – evaluate a model on the test set
 *                               (default: the pre-trained model)
 */

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dataset.hpp"
#include "mlp.h"
#include "config.hpp"

/* ========================================================================== */
/* Parameters                                                                 */
/* ========================================================================== */

const double LEARNING_RATE = 0.01;
const int EPOCHS = 100;                 // Upper limit, early stopping ends earlier
const int TRAINING_SAMPLES = 60000;
const double VALIDATION_FRACTION = 0.2; // 80 % training, 20 % validation

const std::string TRAIN_CSV = "data/mnist_train.csv";
const std::string TEST_CSV = "data/mnist_test.csv";

/** @brief Number of test samples printed in detail before the full evaluation. */
const int PREVIEW_SAMPLES = 20;

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

/**
 * @brief Builds the model file name from the training parameters,
 *        e.g. `models/model_0.01_100_60000_128_64`.
 */
std::string buildModelPath()
{
    std::ostringstream oss;
    oss << "models/model_" << LEARNING_RATE << "_" << EPOCHS << "_" << TRAINING_SAMPLES;
    for (int size : mnist::HIDDEN_LAYERS)
    {
        oss << "_" << size;
    }
    return oss.str();
}

/* ========================================================================== */
/* Training                                                                   */
/* ========================================================================== */

/**
 * @brief Trains a new model with early stopping and saves it.
 */
void train()
{
    Dataset all = loadMnistCsv(TRAIN_CSV, TRAINING_SAMPLES);
    Dataset training, validation;
    splitDataset(all, VALIDATION_FRACTION, training, validation);

    MLP mlp(mnist::INPUT_SIZE, mnist::HIDDEN_LAYERS, mnist::OUTPUT_SIZE, LEARNING_RATE);

    std::cout << "Starting training with " << training.size() << " training samples and "
              << validation.size() << " validation samples." << std::endl;

    mlp.startTraining(training.inputs, training.targets,
                      validation.inputs, validation.targets, EPOCHS);
    std::cout << "Training completed." << std::endl;

    const std::string modelPath = buildModelPath();
    mlp.saveModel(modelPath);
    std::cout << "Model saved to: " << modelPath << std::endl;
}

/* ========================================================================== */
/* Evaluation                                                                 */
/* ========================================================================== */

/**
 * @brief Prints the prediction for the first few test samples.
 */
void previewPredictions(MLP &mlp, const Dataset &test)
{
    std::cout << "\n----- Testing on " << PREVIEW_SAMPLES << " samples -----" << std::endl;

    for (std::size_t i = 0; i < test.size() && i < PREVIEW_SAMPLES; ++i)
    {
        const std::vector<double> output = mlp.forward(test.inputs[i]);
        const int predicted = mnist::argmax(output);
        const int expected = mnist::argmax(test.targets[i]);

        std::cout << "\nSample " << i + 1 << ":" << std::endl;
        std::cout << "Expected Label: " << expected << std::endl;
        std::cout << "Predicted Label: " << predicted << std::endl;
        std::cout << "Correct: " << (predicted == expected ? "YES" : "NO") << std::endl;
        std::cout << "Confidence: " << output[predicted] << std::endl;
    }
}

/**
 * @brief Evaluates the model on the full test set, overall and per digit.
 */
void evaluate(MLP &mlp, const Dataset &test)
{
    std::cout << "\n----- Evaluating on full test set -----" << std::endl;

    int correct = 0;
    std::vector<int> classCorrect(mnist::OUTPUT_SIZE, 0);
    std::vector<int> classTotal(mnist::OUTPUT_SIZE, 0);

    for (std::size_t i = 0; i < test.size(); ++i)
    {
        const int predicted = mnist::argmax(mlp.forward(test.inputs[i]));
        const int expected = mnist::argmax(test.targets[i]);

        classTotal[expected]++;
        if (predicted == expected)
        {
            correct++;
            classCorrect[expected]++;
        }

        if ((i + 1) % 1000 == 0)
        {
            std::cout << "Processed " << i + 1 << " samples..." << std::endl;
        }
    }

    const double accuracy = static_cast<double>(correct) / test.size() * 100.0;

    std::cout << "\n==================== EVALUATION RESULTS ====================" << std::endl;
    std::cout << "Total samples tested: " << test.size() << std::endl;
    std::cout << "Correct predictions: " << correct << std::endl;
    std::cout << "Overall accuracy: " << accuracy << "%" << std::endl;

    std::cout << "\n----- Per-digit accuracy -----" << std::endl;
    for (int digit = 0; digit < mnist::OUTPUT_SIZE; digit++)
    {
        if (classTotal[digit] > 0)
        {
            const double digitAccuracy = static_cast<double>(classCorrect[digit]) / classTotal[digit] * 100.0;
            std::cout << "Digit " << digit << ": " << digitAccuracy << "% ("
                      << classCorrect[digit] << "/" << classTotal[digit] << ")" << std::endl;
        }
    }
    std::cout << "==========================================================" << std::endl;
}

/* ========================================================================== */
/* Main Entry                                                                 */
/* ========================================================================== */

int main(int argc, char *argv[])
{
    const std::string mode = argc > 1 ? argv[1] : "train";

    try
    {
        if (mode == "train")
        {
            train();
        }
        else if (mode == "eval")
        {
            const std::string modelPath = argc > 2 ? argv[2] : mnist::DEFAULT_MODEL;

            // The learning rate is not used for inference
            MLP mlp(mnist::INPUT_SIZE, mnist::HIDDEN_LAYERS, mnist::OUTPUT_SIZE);
            mlp.loadModel(modelPath);
            std::cout << "Model loaded successfully from file: " << modelPath << std::endl;

            const Dataset test = loadMnistCsv(TEST_CSV);
            previewPredictions(mlp, test);
            evaluate(mlp, test);
        }
        else
        {
            std::cerr << "Usage: train [eval [model]]" << std::endl;
            return 1;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
