// 3b1b_backprop.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include "Network.h"

#include <iostream>
#include <chrono>

void Generate(size_t& n1, size_t& n2)
{
	//static size_t callCount = 0;
	//size_t numBits = 2;

	//n1 = callCount % (1u << numBits);
	//n2 = callCount % (1u << numBits);
	//callCount++;

	//size_t numBits = 2;

	//size_t n1, n2;
	//Generate(n1, n2);
	//size_t r = n1 + n2;

	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	input.push_back(float(n1 & 0b1));
	//	n1 = n1 >> 1;
	//}

	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	input.push_back(float(n2 & 0b1));
	//	n2 = n2 >> 1;
	//}


	//for (size_t i = 0; i < (numBits + 1); i++)
	//{
	//	result.push_back(float(r & 0b1));
	//	r = r >> 1;
	//}
}

void GenerateRandom(std::vector<float>& input, std::vector<float>& result)
{
	float x = float((rand() % 1'000'000) - 500'000.0f) / 100.0f;
	float y;
	if (rand() % 2)
	{
		result.push_back(0);
		result.push_back(1);
		y = x*x;
	}
	else
	{
		result.push_back(1);
		result.push_back(0);
		y = 2*x;
	}
	input.push_back(x);
	input.push_back(y);
}

int main()
{
	auto count = std::chrono::high_resolution_clock::now().time_since_epoch().count();
	srand((uint32_t)count);

	Network network{ {2,3,2} };

	float cost = 1.0f;
	size_t batchSize = 200;
	size_t printEveryNBatches = 100;
	float learningRate = 10.0f;

	while (cost > 0.00001f)
	{
		cost = 0.0f;

		for (size_t j = 0; j < printEveryNBatches; j++)
		{
			for (size_t i = 0; i < batchSize; i++)
			{
				std::vector<float> in;
				std::vector<float> out;
				GenerateRandom(in, out);

				cost += network.BackPropagate(in, out);
			}
			network.ConsumeDelta(learningRate);
			learningRate = std::max(0.01f, learningRate * 0.99999f);
		}


		cost /= (batchSize * printEveryNBatches);
		std::cout << "COST: "
			<< cost
			<< " LR: "
			<< learningRate
			<< '\n';
	}

	//network.Serialize();

	while (true)
	{
		float x;
		float y;
		std::cin >> x >> y;

		auto result = network.Propagate({ float(x),float(y) });
		for (float activation : result)
		{
			std::cout << activation << " ";
		}
		std::cout << std::endl;
	}
}