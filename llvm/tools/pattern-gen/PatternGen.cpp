#include "PatternGen.hpp"
#include "../lib/Target/RISCV/RISCVISelLowering.h"
#include "LLVMOverride.hpp"
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/GlobalISel/PatternGen.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <array>
#include <exception>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <type_traits>
#include <unordered_map>
#include <utility>

int optimizeBehavior(llvm::Module *M, std::vector<CDSLInstr> const &Instrs,
                     std::ostream &OstreamIR, PGArgsStruct Args) {
  return runOptPipeline(M, Args.Is64Bit, Args.Mattr, Args.OptLevel, OstreamIR);
}

int generatePatterns(llvm::Module *M, std::vector<CDSLInstr> const &Instrs,
                     std::ostream &Ostream, PGArgsStruct Args) {
  // All other code in this file is called during code generation
  // by the LLVM pipeline. We thus "pass" arguments as globals.
  llvm::PatternGenArgs::OutStream = &Ostream;
  llvm::PatternGenArgs::Args = Args;
  llvm::PatternGenArgs::Instrs = &Instrs;

  if (!Args.Predicates.empty())
    Ostream << "let Predicates = [" << Args.Predicates << "] in {\n\n";

  int Rv = runPatternGenPipeline(M, Args.Is64Bit, Args.Mattr);

  if (!Args.Predicates.empty())
    Ostream << "}\n";

  llvm::PatternGenArgs::OutStream = nullptr;
  llvm::PatternGenArgs::Args = PGArgsStruct();
  llvm::PatternGenArgs::Instrs = nullptr;

  return Rv;
}
