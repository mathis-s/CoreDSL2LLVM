#include "PatternGen.hpp"
#include "lib/InstrInfo.hpp"
#include "LLVMOverride.hpp"
#include "../lib/Target/RISCV/RISCVISelLowering.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/Support/Casting.h"
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

static std::ostream* outStream = nullptr;
static std::vector<CDSLInstr> const* cdslInstrs;
static CDSLInstr const* curInstr = nullptr;
static std::string* extName = nullptr;

using namespace llvm;
using SVT = llvm::MVT::SimpleValueType;

int GeneratePatterns(llvm::Module* M, std::vector<CDSLInstr> const& instrs, std::ostream& ostream, std::string extName, size_t opt_level, std::string mattr)
{
    // All other code in this file is called during code generation
    // by the LLVM pipeline. We thus "pass" arguments as globals in this TU.
    outStream = &ostream;
    cdslInstrs = &instrs;
    ::extName = &extName;

    int rv = RunPatternGenPipeline(M, mattr, opt_level);

    outStream = nullptr;
    cdslInstrs = nullptr;
    ::extName = nullptr;

    return rv;
}
