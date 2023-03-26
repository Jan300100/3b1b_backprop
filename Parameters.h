#pragma once
#include <vector>
#include "Serializable.h"


struct Parameters : public Serializable
{
	Parameters() = default;
	Parameters(size_t numBiases, size_t numWeights, bool zeroInit = true);
	Parameters(std::string serializedParams);
	Parameters& operator+=(const Parameters& other);
	Parameters& operator*=(float other);
	
	void Clear();
	std::vector<float> weights;
	std::vector<float> biases;



	// Inherited via Serializable
	virtual std::string Serialize() override;
	virtual void Deserialize(const std::string& inString) override;
};

