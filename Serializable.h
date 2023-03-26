#pragma once

#include <string>

class Serializable
{
public:
	virtual std::string Serialize() = 0;
	virtual void Deserialize(const std::string&) = 0;
};