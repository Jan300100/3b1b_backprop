#include "Network.h"
#include "Layer.h"
#include "ActivationFunctions.h"

Network::Network(std::vector<size_t> neuronsPerLayer, bool zeroInit)
{
	m_initialLayer = new InitialLayer{ neuronsPerLayer[0] };

	m_numStored = 1;

	m_layers.push_back(m_initialLayer);
	m_storedDelta.push_back(Parameters{ m_initialLayer->GetNumNeurons(), m_initialLayer->GetNumWeightsToPrevious(), true });

	for (size_t i = 1; i < neuronsPerLayer.size(); i++)
	{
		if (i < (neuronsPerLayer.size() - 1))
		{
			m_layers.push_back(new Layer<Sigmoid>{ neuronsPerLayer[i], m_layers.back() });
		}
		else
		{
			m_layers.push_back(new Layer<Sigmoid>{ neuronsPerLayer[i], m_layers.back() });
		}

		m_storedDelta.push_back(Parameters{ m_layers.back()->GetNumNeurons(), m_layers.back()->GetNumWeightsToPrevious() ,true });
	}
}

Network::~Network()
{
	for (LayerBase* l : m_layers)
	{
		delete l;
	}
}

void Network::StoreDelta(const std::vector<Parameters>& other)
{
	if (m_storedDelta.size() == other.size())
	{
		// no need to update first layer
		for (size_t i = 1; i < m_storedDelta.size(); i++)
		{
			m_storedDelta[i] += other[i];
		}

		m_numStored++;
	}
}

void Network::ConsumeDelta(float learningRate)
{
	if (m_layers.size() == m_storedDelta.size())
	{
		// no need to update first layer
		for (size_t i = 1; i < m_layers.size(); i++)
		{
			if (m_layers[i]->m_params.biases.size() == m_storedDelta[i].biases.size())
			{
				for (size_t j = 0; j < m_layers[i]->m_params.biases.size(); j++)
				{
					m_layers[i]->m_params.biases[j] -= (learningRate * m_storedDelta[i].biases[j]) / m_numStored;
					m_storedDelta[i].biases[j] = 0.0f;
				}
			}
			if (m_layers[i]->m_params.weights.size() == m_storedDelta[i].weights.size())
			{
				for (size_t j = 0; j < m_layers[i]->m_params.weights.size(); j++)
				{
					m_layers[i]->m_params.weights[j] -= (learningRate * m_storedDelta[i].weights[j]) / m_numStored;
					m_storedDelta[i].weights[j] = 0.0f;
				}
			}
		}

		m_numStored = 0;
	}
}

float Network::CalculateCost(std::vector<float> inputActivation, std::vector<float> preferredOutput)
{
	const std::vector<float>& result = Propagate(inputActivation);

	float cost = 0.0f;
	if (result.size() == preferredOutput.size())
	{
		for (size_t i = 0; i < result.size(); i++)
		{
			cost += powf(result[i] - preferredOutput[i], 2.0f);
		}
		return cost;
	}

	return std::numeric_limits<float>().infinity();
}

std::vector<float> Network::Propagate(std::vector<float> inputActivation)
{
	m_initialLayer->StartPropagation(inputActivation);
	return m_layers.back()->m_activations;
}

float Network::BackPropagate(std::vector<float> inputActivation, std::vector<float> preferredOutput)
{
	// propagate forwards
	float cost = CalculateCost(inputActivation, preferredOutput);

	// create empty network with same dimensions to store deltas. 
	std::vector<Parameters> deltaParameters;
	deltaParameters.resize(m_layers.size());
	deltaParameters.front() = Parameters{ m_initialLayer->GetNumNeurons(), m_initialLayer->GetNumWeightsToPrevious(), true };

	// propagate backwards
	LayerBase* layer = m_layers.back();
	std::vector<float> prevLayerCostDeltas;
	prevLayerCostDeltas.resize(layer->m_numNeurons);
	for (size_t i = 0; i < layer->m_numNeurons; i++)
	{
		float act = layer->m_activations[i];
		float y = preferredOutput[i];

		prevLayerCostDeltas[i] = 2 * (act - y);
	}

	size_t layerIndex = m_layers.size() - 1;
	while (layer->m_previousLayer != nullptr)
	{
		Parameters& deltaLayer = deltaParameters[layerIndex] = Parameters(layer->GetNumNeurons(), layer->GetNumWeightsToPrevious(), true);

		std::vector<float> currentCostDeltas = prevLayerCostDeltas;

		for (size_t i = 0; i < layer->m_numNeurons; i++)
		{
			//calc bias nudge
			float z = layer->m_preProcessedActivations[i];
			float actFuncDeriv = layer->m_activationFunction->ExecuteDerivative(z);

			deltaLayer.biases[i] = actFuncDeriv * currentCostDeltas[i];

			//calc weight nudge (bias nudge * A(L-1)) for each weight
			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
			size_t weightIndexStart = i * weightsPerNeuron;
			for (size_t j = 0; j < layer->m_previousLayer->m_numNeurons; j++)
			{
				deltaLayer.weights[weightIndexStart + j] = layer->m_previousLayer->m_activations[j] * deltaLayer.biases[i];
			}
		}

		// setup for next layer
		prevLayerCostDeltas.clear();
		prevLayerCostDeltas.resize(layer->m_previousLayer->m_numNeurons, 0.0f);

		for (size_t i = 0; i < layer->m_previousLayer->m_numNeurons; i++)
		{
			// for each outgoing weight from neuron[j]
			// calc weight * weightDestNeuron.biasDelta;
			// sum it. save in prevLayerCostDeltas[j]
			size_t relevantWeightIndex = i;
			size_t weightsPerNeuron = layer->m_previousLayer->m_numNeurons;
			for (size_t j = 0; j < layer->m_numNeurons; j++)
			{
				size_t weightIdx = weightsPerNeuron * j + relevantWeightIndex;
				prevLayerCostDeltas[i] += layer->m_params.weights[weightIdx] * deltaLayer.biases[j];
			}
		}
		layer = layer->m_previousLayer;
		layerIndex--;
	}

	// update weights and biases
	StoreDelta(deltaParameters);

	return cost;
}
