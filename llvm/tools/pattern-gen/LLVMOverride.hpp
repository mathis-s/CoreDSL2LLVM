#pragma once
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"

int RunPatternGenPipeline(llvm::Module* M, std::string mattr, llvm::CodeGenOptLevel optLevel, std::ostream &irOut);
