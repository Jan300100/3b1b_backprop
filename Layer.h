#pragma once
#include <vector>
#include <memory>
#include "Parameters.h"
#include "ActivationFunctions.h"

class Layer
{
public:
	Layer(size_t numNeurons, ActFunc::Base* func, Layer* previousLayer = nullptr);
	virtual ~Layer() = default;

	void SetParams(const Parameters& params);

	size_t GetNumNeurons() const
	{
		return m_numNeurons;
	}

	size_t GetNumWeightsToPrevious() const
	{
		return m_params.weights.size();
	}

protected:
	void Propagate();

protected:
	ActFunc::Base* m_activationFunction;

	Layer* m_nextLayer = nullptr;
	Layer* m_previousLayer = nullptr;

	Parameters m_params;

	size_t m_numNeurons;
	std::vector<float> m_preProcessedActivations;
	std::vector<float> m_activations;

	friend class Network;
};

class InitialLayer : public Layer
{
public:
	InitialLayer(size_t numNeurons);
	virtual ~InitialLayer() = default;
	void StartPropagation(std::vector<float> inputActivation);
};
