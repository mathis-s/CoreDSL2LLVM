#pragma once
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/Support/CodeGen.h"
#include <llvm/IR/Module.h>

struct PGArgsStruct
{
    std::string Mattr;
    llvm::CodeGenOptLevel OptLevel;
    std::string Predicates;
    bool is64Bit;
};

int OptimizeBehavior(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostreamIR, PGArgsStruct args);
int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, PGArgsStruct args);
//void PrintPattern(llvm::SelectionDAG& DAG);
