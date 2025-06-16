/* ========================================================================== */
/* dataset.hpp                                                                */
/* ========================================================================== */
/**
 * @file      dataset.hpp
 * @author    wehrberger
 * @date      16.06.2025
 *
 * @brief     Loading of the MNIST dataset in CSV format.
 *
 * Every row has the form `label,pixel0,...,pixel783`.
 * Source: https://pjreddie.com/projects/mnist-in-csv/
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

/** @brief Normalised images with their one-hot encoded labels. */
struct Dataset
{
    std::vector<std::vector<double>> inputs;   /**< 784 pixels in [0, 1]. */
    std::vector<std::vector<double>> targets;  /**< One-hot labels. */

    std::size_t size() const { return inputs.size(); }
};

/**
 * @brief Loads up to @p maxSamples rows from an MNIST CSV file.
 *
 * @throws std::runtime_error if the file cannot be opened or a row is malformed.
 */
Dataset loadMnistCsv(const std::string &path, std::size_t maxSamples = SIZE_MAX);

/**
 * @brief Splits a dataset into a training and a validation part.
 *
 * The last @p validationFraction of the samples becomes the validation set.
 */
void splitDataset(const Dataset &all, double validationFraction,
                  Dataset &training, Dataset &validation);

/**
 * @brief One-hot encodes a digit label.
 *
 * @throws std::runtime_error if the label is out of range.
 */
std::vector<double> oneHotEncode(int label);
