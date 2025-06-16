/* ========================================================================== */
/* dataset.cpp                                                                */
/* ========================================================================== */
/**
 * @file      dataset.cpp
 * @author    wehrberger
 * @date      16.06.2025
 *
 * @brief     Implementation of the MNIST CSV loader.
 */

#include "dataset.hpp"
#include "config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

/**
 * @brief Parses one CSV row into label and pixels.
 *
 * @return false for an empty line (end of file).
 */
static bool parseRow(const std::string &line, int &label, std::vector<unsigned char> &pixels)
{
    if (line.empty())
    {
        return false;
    }

    std::stringstream ss(line);
    std::string token;
    std::vector<int> values;
    values.reserve(mnist::INPUT_SIZE + 1);
    while (std::getline(ss, token, ','))
    {
        values.push_back(std::stoi(token));
    }

    if (values.size() != mnist::INPUT_SIZE + 1) // 1 label + 784 pixels
    {
        throw std::runtime_error("Invalid row format in CSV file.");
    }

    label = values[0];
    pixels.assign(values.begin() + 1, values.end());
    return true;
}

Dataset loadMnistCsv(const std::string &path, std::size_t maxSamples)
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open the CSV file: " + path);
    }

    Dataset data;
    std::string line;
    int label;
    std::vector<unsigned char> pixels;

    while (data.size() < maxSamples && std::getline(file, line))
    {
        if (!parseRow(line, label, pixels))
        {
            continue;
        }
        data.inputs.push_back(mnist::normalizePixels(pixels));
        data.targets.push_back(oneHotEncode(label));
    }
    return data;
}

void splitDataset(const Dataset &all, double validationFraction,
                  Dataset &training, Dataset &validation)
{
    const std::size_t validationSize = static_cast<std::size_t>(all.size() * validationFraction);
    const std::size_t trainingSize = all.size() - validationSize;

    training.inputs.assign(all.inputs.begin(), all.inputs.begin() + trainingSize);
    training.targets.assign(all.targets.begin(), all.targets.begin() + trainingSize);
    validation.inputs.assign(all.inputs.begin() + trainingSize, all.inputs.end());
    validation.targets.assign(all.targets.begin() + trainingSize, all.targets.end());
}

std::vector<double> oneHotEncode(int label)
{
    if (label < 0 || label >= mnist::OUTPUT_SIZE)
    {
        throw std::runtime_error("Label out of range for one-hot encoding.");
    }
    std::vector<double> encoded(mnist::OUTPUT_SIZE, 0.0);
    encoded[label] = 1.0;
    return encoded;
}
