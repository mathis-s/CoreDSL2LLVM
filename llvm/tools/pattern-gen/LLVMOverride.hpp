#pragma once
#include "llvm/IR/Module.h"
#include "llvm/Support/CodeGen.h"

int runOptPipeline(llvm::Module* M, bool Is64Bit, std::string Mattr, llvm::CodeGenOptLevel OptLevel, std::ostream &IrOut);
int runPatternGenPipeline(llvm::Module* M, bool Is64Bit, std::string Mattr);
