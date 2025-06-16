/* ========================================================================== */
/* mlp.h                                                                      */
/* ========================================================================== */
/**
 * @file      mlp.h
 * @author    wehrberger
 * @date      16.06.2025
 *
 * @brief     Multi-layer perceptron with sigmoid hidden layers and a softmax
 *            output layer, trained by backpropagation.
 *
 * Built as a shared library (mlp.dll) and used by the MNIST training app.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "perceptron.h"

#ifdef _WIN32
#ifdef MLP_EXPORT
#define MLP_API __declspec(dllexport)
#else
#define MLP_API __declspec(dllimport)
#endif
#else
#define MLP_API
#endif

class MLP_API MLP
{
public:
    /**
     * Constructor for a network with multiple hidden layers.
     * @param inputSize Number of input neurons.
     * @param hiddenSizes A vector containing the size of each hidden layer.
     * @param outputSize Number of output neurons.
     * @param learningRate Learning rate for training.
     */
    MLP(int inputSize, const std::vector<int> &hiddenSizes,
        int outputSize, double learningRate = 0.1);
    ~MLP();

    MLP(const MLP &) = delete;
    MLP &operator=(const MLP &) = delete;

    /**
     * @brief Forward pass.
     * @return Softmax probabilities of the output layer.
     */
    std::vector<double> forward(const std::vector<double> &inputs);

    /**
     * @brief Trains the network with early stopping on validation accuracy.
     *
     * @param epochs              Maximum number of epochs.
     * @param patience            Epochs without improvement before stopping.
     * @param minimalImprovement  Required increase of validation accuracy.
     */
    void startTraining(const std::vector<std::vector<double>> &trainingInputs,
                       const std::vector<std::vector<double>> &trainingTargets,
                       const std::vector<std::vector<double>> &validationInputs,
                       const std::vector<std::vector<double>> &validationTargets,
                       int epochs, int patience = 5,
                       double minimalImprovement = 0.001);

    /** @brief Saves all weights and biases to a binary file. */
    void saveModel(const std::string &filename);

    /** @brief Loads a model written by saveModel(). */
    void loadModel(const std::string &filename);

    /** @brief Fraction of correctly classified samples (targets one-hot). */
    double computeAccuracy(const std::vector<std::vector<double>> &inputs,
                           const std::vector<std::vector<double>> &targets);

private:
    // PIMPL-style internal implementation
    struct Layers;
    std::unique_ptr<Layers> m_Layers;

    // A backpropagation training step
    void train(const std::vector<double> &inputs,
               const std::vector<double> &targets);

    // Compute the output of a vector of Perceptron (a layer), given the input.
    std::vector<double> computeLayerOutput(const std::vector<Perceptron> &layer,
                                           const std::vector<double> &inputs,
                                           bool skipActivation = false);

    // Apply softmax to a vector of values
    std::vector<double> applySoftmax(const std::vector<double> &inputs);

    // Get predicted class (index of maximum value)
    int getPredictedClass(const std::vector<double> &output);
};
