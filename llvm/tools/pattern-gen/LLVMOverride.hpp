#pragma once
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"

int RunOptPipeline(llvm::Module* M, bool is64Bit, std::string mattr, llvm::CodeGenOptLevel optLevel, std::ostream &irOut);
int RunPatternGenPipeline(llvm::Module* M, bool is64Bit, std::string mattr);
