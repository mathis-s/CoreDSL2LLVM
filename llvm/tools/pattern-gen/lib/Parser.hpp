#pragma once
#include "TokenStream.hpp"
#include "InstrInfo.hpp"
#include <llvm/IR/Module.h>

std::vector<CDSLInstr> ParseCoreDSL2(TokenStream& ts, llvm::Module* mod);
