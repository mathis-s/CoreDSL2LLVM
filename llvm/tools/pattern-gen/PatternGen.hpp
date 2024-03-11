#pragma once
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/CodeGen.h"
#include <llvm/IR/Module.h>

struct PGArgsStruct
{
    std::string ExtName;
    std::string Mattr;
    llvm::CodeGenOptLevel OptLevel;
    std::string Predicates;
};

int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, std::ostream& ostreamIR, PGArgsStruct args);
//void PrintPattern(llvm::SelectionDAG& DAG);
