#pragma once
#include <vector>
#include "Parameters.h"

class ActivationFunction;

class LayerBase
{
public:
	LayerBase(size_t numNeurons, ActivationFunction* func, LayerBase* previousLayer = nullptr);
	virtual ~LayerBase() = default;


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
	ActivationFunction* m_activationFunction;

	LayerBase* m_nextLayer = nullptr;
	LayerBase* m_previousLayer = nullptr;

	Parameters m_params;

	size_t m_numNeurons;
	std::vector<float> m_preProcessedActivations;
	std::vector<float> m_activations;

	friend class Network;
};

class InitialLayer : public LayerBase
{
public:
	InitialLayer(size_t numNeurons);
	virtual ~InitialLayer() = default;
	void StartPropagation(std::vector<float> inputActivation);
};

template <typename ActFunc>
class Layer : public LayerBase
{
public:
	Layer(size_t numNeurons, LayerBase* previousLayer = nullptr);
	virtual ~Layer();
private:
	ActFunc* m_actFunc;
};

template<typename ActFunc>
inline Layer<ActFunc>::Layer(size_t numNeurons, LayerBase* previousLayer)
	:LayerBase(numNeurons, new ActFunc{}, previousLayer)
{
	// take ownership of activationFunc;
	m_actFunc = dynamic_cast<ActFunc*>(m_activationFunction);
}

template<typename ActFunc>
inline Layer<ActFunc>::~Layer()
{
	delete m_actFunc;
}