#include "PatternGen.hpp"
#include "../lib/Target/RISCV/RISCVISelLowering.h"
#include "LLVMOverride.hpp"
#include "lib/InstrInfo.hpp"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/Errc.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/CodeGen/GlobalISel/PatternGen.h"
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

int GeneratePatterns(llvm::Module *M, std::vector<CDSLInstr> const &instrs,
                     std::ostream &ostream, std::ostream &ostreamIR, std::string extName,
                     llvm::CodeGenOptLevel optLevel, std::string mattr) {
  // All other code in this file is called during code generation
  // by the LLVM pipeline. We thus "pass" arguments as globals.
  llvm::PatternGenArgs::OutStream = &ostream;
  llvm::PatternGenArgs::ExtName = &extName;
  llvm::PatternGenArgs::Instrs = &instrs;

  int rv = RunPatternGenPipeline(M, mattr, optLevel, ostreamIR);

  llvm::PatternGenArgs::OutStream = nullptr;
  llvm::PatternGenArgs::ExtName = nullptr;
  llvm::PatternGenArgs::Instrs = nullptr;

  return rv;
}
