#pragma once
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/SelectionDAG.h"
#include <llvm/IR/Module.h>

int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, std::string extName, size_t opt_level, std::string mattr);
//void PrintPattern(llvm::SelectionDAG& DAG);
