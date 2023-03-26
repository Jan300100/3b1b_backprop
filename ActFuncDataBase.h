#pragma once
#include <string>

#include "ActivationFunctions.h"

class ActFuncDataBase
{

public:
	ActivationFunction* FindActFunc(const std::string& name);
};
