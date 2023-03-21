#pragma once
#include <algorithm>


class ActivationFunction
{
public:
	virtual float Execute(float x) const = 0;
	virtual float ExecuteDerivative(float x) const = 0;
};


class Sigmoid : public ActivationFunction
{
public:
	virtual float Execute(float x) const override
	{
		return 1.0f / (1.0f + expf(-x));
	}
	virtual float ExecuteDerivative(float x) const override
	{
		float s = Execute(x);
		return s * (1.0f - s);
	}
};

class None : public ActivationFunction
{
public:
	virtual float Execute(float x) const override
	{
		return x;
	}
	virtual float ExecuteDerivative(float x) const override
	{
		return 1;
	}
};

class ReLu : public ActivationFunction
{
public:
	virtual float Execute(float x) const override
	{
		return std::max(0.0f, x);
	}
	virtual float ExecuteDerivative(float x) const override
	{
		return x > 0;
	}
};

class LeakyReLu : public ActivationFunction
{
	static constexpr float leakySlope = 0.1f;
public:
	virtual float Execute(float x) const override
	{
		if (x >= 0.0f)
		{
			return x;
		}
		else
		{
			return x * leakySlope;
		}

		return std::max(0.0f, x);
	}
	virtual float ExecuteDerivative(float x) const override
	{
		return x >= 0 ? 1 : leakySlope;
	}
};