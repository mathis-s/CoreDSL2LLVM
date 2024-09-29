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
    bool Is64Bit;
};

int optimizeBehavior(llvm::Module* M, std::vector<CDSLInstr> const& Instrs, std::ostream& OstreamIR, PGArgsStruct Args);
int generatePatterns(llvm::Module* M, std::vector<CDSLInstr> const& Instrs, std::ostream& Ostream, PGArgsStruct Args);
//void PrintPattern(llvm::SelectionDAG& DAG);
