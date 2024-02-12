#pragma once
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/SelectionDAG.h"
#include <llvm/IR/Module.h>

int OptimizeBehavior(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostreamIR, std::string extName, llvm::CodeGenOptLevel optLevel, std::string mattr);
int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, std::string extName, std::string mattr);
//void PrintPattern(llvm::SelectionDAG& DAG);
