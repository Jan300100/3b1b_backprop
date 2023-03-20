#pragma once
#include <vector>
#include "Parameters.h"

class LayerBase;
class InitialLayer;
class ActivationFunction;

class Network
{
public:
	Network(std::vector<size_t> neuronsPerLayer, bool zeroInit = false);
	Network(Network&) = delete;
	Network(Network&&) = delete;
	Network& operator=(Network&) = delete;
	Network& operator=(Network&&) = delete;
	~Network();

	float CalculateCost(std::vector<float> inputActivation, std::vector<float> preferredOutput);
	std::vector<float> Propagate(std::vector<float> inputActivation);

	
	float BackPropagate(std::vector<float> inputActivation, std::vector<float> preferredOutput);

	void ConsumeDelta(float learningRate);

private:
	void StoreDelta(const std::vector<Parameters>& other);

private:
	InitialLayer* m_initialLayer;

	std::vector<LayerBase*> m_layers;
	std::vector<Parameters> m_storedDelta;

	size_t m_numStored = 0;
};