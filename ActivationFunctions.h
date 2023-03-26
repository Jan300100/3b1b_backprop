#pragma once

class ActivationFunction
{
public:
	virtual std::string GetName() const = 0;
	virtual float Execute(float x) const = 0;
	virtual float ExecuteDerivative(float x) const = 0;
};

class Sigmoid : public ActivationFunction
{
	static constexpr const char* Name{"Sigmoid"};
public:
	virtual std::string GetName() const
	{
		return Name;
	}

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
	static constexpr const char* Name{ "None" };
public:
	virtual std::string GetName() const
	{
		return Name;
	}

	virtual float Execute(float x) const override
	{
		return x;
	}

	virtual float ExecuteDerivative(float x) const override
	{
		return 1;
	}
};

class ReLU : public ActivationFunction
{
	static constexpr const char* Name{ "ReLU" };
public:
	virtual std::string GetName() const
	{
		return Name;
	}

	virtual float Execute(float x) const override
	{
		return std::max(0.0f, x);
	}

	virtual float ExecuteDerivative(float x) const override
	{
		return x > 0;
	}
};

class LeakyReLU : public ActivationFunction
{
	static constexpr float k_leakySlope = 0.1f;
	static constexpr const char* Name{ "LeakyReLU" };
public:
	virtual std::string GetName() const
	{
		return Name;
	}

	virtual float Execute(float x) const override
	{
		if (x >= 0.0f)
		{
			return x;
		}
		else
		{
			return x * k_leakySlope;
		}

		return std::max(0.0f, x);
	}

	virtual float ExecuteDerivative(float x) const override
	{
		return x >= 0 ? 1 : k_leakySlope;
	}
};