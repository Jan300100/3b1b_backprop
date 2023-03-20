#include "Layer.h"
#include "ActivationFunctions.h"

LayerBase::LayerBase(size_t numNeurons, ActivationFunction* func, LayerBase* previousLayer)
	: m_numNeurons{numNeurons}
	, m_activationFunction{func}
	, m_previousLayer{previousLayer}
{
	m_preProcessedActivations.resize(numNeurons);
	m_activations.resize(numNeurons);

	size_t numWeightsToEachNeuron = 0;
	if (previousLayer)
	{
		previousLayer->m_nextLayer = this;
		numWeightsToEachNeuron = previousLayer->m_numNeurons;
	}

	m_params = Parameters{ numNeurons, numWeightsToEachNeuron * numNeurons, false };
}

void LayerBase::Propagate()
{
	// We need previousLayer to work
	if (m_previousLayer != nullptr)
	{
		// For each of my neurons
		size_t weightsPerNeuron = m_previousLayer->m_activations.size();
		for (size_t i = 0; i < m_activations.size(); i++)
		{
			size_t startWeight = i * weightsPerNeuron;

			float activation = 0.0f;
			for (size_t j = 0; j < m_previousLayer->m_activations.size(); j++)
			{
				float prevActivation = m_previousLayer->m_activations[j];
				float weight = m_params.weights[startWeight + j];
				activation += weight * prevActivation;
			}
			activation += m_params.biases[i];

			m_preProcessedActivations[i] = activation;
			if (m_activationFunction != nullptr)
			{
				activation = m_activationFunction->Execute(activation);
			}

			m_activations[i] = activation;
		}
	}

	// propagate
	if (m_nextLayer)
	{
		m_nextLayer->Propagate();
	}
}

InitialLayer::InitialLayer(size_t numNeurons)
	:LayerBase{numNeurons, nullptr}
{
}

void InitialLayer::StartPropagation(std::vector<float> inputActivation)
{
	if (inputActivation.size() == m_numNeurons)
	{
		m_activations = std::move(inputActivation);
		m_preProcessedActivations = m_activations;
		Propagate();
	}
}
