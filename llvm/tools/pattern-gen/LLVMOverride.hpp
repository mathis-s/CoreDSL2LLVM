#pragma once
#include "llvm/IR/Module.h"

int RunPatternGenPipeline(llvm::Module* M, std::string mattr, size_t opt_level);
