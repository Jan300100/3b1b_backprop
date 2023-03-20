#pragma once
#include <vector>

struct Parameters
{
	Parameters() = default;
	Parameters(size_t numBiases, size_t numWeights, bool zeroInit = true);
	Parameters& operator+=(const Parameters& other);


	std::vector<float> weights;
	std::vector<float> biases;
};