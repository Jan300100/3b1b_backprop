#include "Parameters.h"


Parameters::Parameters(size_t numBiases, size_t numWeights, bool zeroInit)
{
	biases.resize(numBiases, 0.0f);
	weights.resize(numWeights, 0.0f);

	if (!zeroInit)
	{
		for (size_t i = 0; i < numWeights; i++)
		{
			weights[i] = (((rand() % 2000) / 1000.f) - 1.0f);
		}
	}
}

Parameters& Parameters::operator+=(const Parameters& other)
{
	if (biases.size() == other.biases.size())
	{
		for (size_t j = 0; j < biases.size(); j++)
		{
			biases[j] += other.biases[j];
		}
	}

	if (weights.size() == other.weights.size())
	{
		for (size_t j = 0; j < weights.size(); j++)
		{
			weights[j] += other.weights[j];
		}
	}

	return *this;
}

Parameters& Parameters::operator*=(float other)
{
	for (size_t j = 0; j < biases.size(); j++)
	{
		biases[j] *= other;
	}
	for (size_t j = 0; j < weights.size(); j++)
	{
		weights[j] *= other;
	}

	return *this;
}

void Parameters::Clear()
{
	(*this) *= 0.0f;
}
