#include "Parameters.h"
#include <sstream>
#include <cassert>


Parameters::Parameters(size_t numBiases, size_t numWeights, bool zeroInit)
{
	biases.resize(numBiases, 0.0f);
	weights.resize(numWeights, 0.0f);

	if (!zeroInit)
	{
		for (size_t i = 0; i < numBiases; i++)
		{
			biases[i] = 0.1f;
		}

		// RANDOM
		for (size_t i = 0; i < numWeights; i++)
		{
			weights[i] = (((rand() % 2000) / 1000.f) - 1.0f);
		}

		// XAVIER


		// HE
		
		
		// ...
	}
}

Parameters::Parameters(std::string serializedParams)
{
	Deserialize(serializedParams);
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

std::string Parameters::Serialize()
{
	std::ostringstream out{};

	char delim = ' ';
	out << biases.size() << delim;
	for (size_t i = 0; i < biases.size(); i++)
	{
		out << biases[i] << delim;
	}

	out << weights.size() << delim;
	for (size_t i = 0; i < weights.size(); i++)
	{
		out << weights[i] << delim;
	}
	
	out << "end";

	return out.str();
}

void Parameters::Deserialize(const std::string& inString)
{
	std::istringstream input{ inString };

	size_t numBiases;
	input >> numBiases;

	biases.clear();
	for (size_t i = 0; i < numBiases; i++)
	{
		float bias;
		input >> bias;
		biases.push_back(bias);
	}

	size_t numWeights;
	input >> numWeights;

	weights.clear();
	for (size_t i = 0; i < numWeights; i++)
	{
		float weight;
		input >> weight;
		weights.push_back(weight);
	}

	std::string end;
	input >> end;

	assert(end == "end");
}
