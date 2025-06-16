/* ========================================================================== */
/* config.hpp                                                                 */
/* ========================================================================== */
/**
 * @file      config.hpp
 * @author    wehrberger
 * @date      16.06.2025
 *
 * @brief     Network configuration and helpers for training and evaluation.
 */

#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace mnist
{

/* ========================================================================== */
/* Network configuration                                                      */
/* ========================================================================== */

/** @brief Image width / height in pixels. */
inline constexpr int IMAGE_SIZE = 28;

/** @brief Number of input neurons (28 × 28 pixels). */
inline constexpr int INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE;

/** @brief Number of output neurons (digits 0-9). */
inline constexpr int OUTPUT_SIZE = 10;

/** @brief Neurons per hidden layer. */
inline const std::vector<int> HIDDEN_LAYERS = {128, 64};

/** @brief Pre-trained model shipped with the repository. */
inline const std::string DEFAULT_MODEL = "models/model_0.01_100_60000_128_64";

/* ========================================================================== */
/* Helpers                                                                    */
/* ========================================================================== */

/**
 * @brief Converts pixel values [0, 255] to doubles in [0, 1].
 */
inline std::vector<double> normalizePixels(const std::vector<unsigned char> &pixels)
{
    std::vector<double> normalized;
    normalized.reserve(pixels.size());
    for (unsigned char p : pixels)
    {
        normalized.push_back(static_cast<double>(p) / 255.0);
    }
    return normalized;
}

/**
 * @brief Returns the index of the largest value (the predicted digit).
 */
inline int argmax(const std::vector<double> &values)
{
    return static_cast<int>(std::max_element(values.begin(), values.end()) - values.begin());
}

} // namespace mnist
