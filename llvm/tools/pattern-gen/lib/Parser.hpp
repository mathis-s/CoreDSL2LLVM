#pragma once
#include "InstrInfo.hpp"
#include "TokenStream.hpp"
#include <llvm/IR/Module.h>

std::vector<CDSLInstr> ParseCoreDSL2(TokenStream &ts, bool is64Bit,
                                     llvm::Module *mod, bool NoExtend);
