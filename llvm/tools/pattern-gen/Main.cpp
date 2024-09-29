#include <cstdio>
#include <ctype.h>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>

#include "PatternGen.hpp"
#include "lib/InstrInfo.hpp"
#include "lib/Lexer.hpp"
#include "lib/Parser.hpp"
#include "lib/Token.hpp"
#include "lib/TokenStream.hpp"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

using namespace llvm;

static cl::OptionCategory ToolOptions("Tool Options");
static cl::OptionCategory ViewOptions("View Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(ToolOptions), cl::init("-"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::init("-"), cl::cat(ToolOptions),
                                           cl::value_desc("filename"));

static cl::opt<std::string>
    InputLanguage("x", cl::desc("Input language ('cdsl' or 'll')"),
                  cl::cat(ToolOptions));

static cl::opt<bool> Force("f", cl::desc("Ignore parser errors."),
                           cl::cat(ToolOptions));
static cl::opt<bool> SkipFmt("skip-formats",
                             cl::desc("Skip tablegen formats step."),
                             cl::cat(ToolOptions));
static cl::opt<bool> SkipPat("skip-patterns",
                             cl::desc("Skip pattern-gen step."),
                             cl::cat(ToolOptions));
static cl::opt<bool> SkipVerify("skip-verify",
                                cl::desc("Skip verification step."),
                                cl::cat(ToolOptions));
static cl::opt<bool> PrintIR("print-ir", cl::desc("Print LLVM-IR module."),
                             cl::cat(ToolOptions));
static cl::opt<bool> NoExtend(
    "no-extend",
    cl::desc("Do not apply CDSL typing rules (Use C-like type inference)."),
    cl::cat(ToolOptions));
// static cl::opt<std::string>
//     Mattr("mattr2", cl::desc("Target specific attributes"),
//           cl::value_desc("a1,+a2,-a3,..."), cl::cat(ToolOptions),
//           // cl::init("+m,+fast-unaligned-access,+xcvalu,+xcvsimd"));
//           cl::init("+m,+fast-unaligned-access"));

static cl::opt<int> XLen("riscv-xlen", cl::desc("RISC-V XLEN (32 or 64 bit)"),
                         cl::init(32));

// Determine optimization level.
static cl::opt<char>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O3')"),
             cl::cat(ToolOptions), cl::init('3'));

static cl::opt<std::string> Predicates(
    "p", cl::desc("Predicate(s) used for instructions in output TableGen"),
    cl::cat(ToolOptions), cl::init("HasVendorXCValu"));

#include <iostream>
namespace fs = std::filesystem;

static auto getOutStreams(std::string SrcPath, std::string DestPath,
                          bool EmitLL) {
  fs::path OutPath{DestPath};

  fs::path InPath{SrcPath};
  fs::path BasePath = InPath.parent_path() / InPath.stem();

  std::string NewExt = ".td";
  if (OutPath.compare("-") != 0) {
    BasePath = OutPath.parent_path() / OutPath.stem();
    NewExt = OutPath.extension();
  }
  // TODO: allow .td in out path
  std::string IrPath = "/dev/null";
  std::string FmtPath = "/dev/null";
  std::string PatPath = "/dev/null";
  if (EmitLL) {
    IrPath = BasePath.string() + ".ll";
  }
  if (!SkipFmt) {
    FmtPath = BasePath.string() + "InstrFormat" + NewExt;
  }
  if (!SkipPat) {
    PatPath = BasePath.string() + NewExt;
  }

  return std::make_tuple(std::ofstream(IrPath), std::ofstream(FmtPath),
                         std::ofstream(PatPath));
}

int main(int argc, char **argv) {
  cl::HideUnrelatedOptions({&ToolOptions, &ViewOptions});
  cl::ParseCommandLineOptions(argc, argv, "CoreDSL2LLVM Pattern Gen");
  if (argc <= 1) {
    fprintf(stderr, "usage: %s <SOURCE FILE> LLVM-ARGS...\n", argv[0]);
    return -1;
  }

  // const char* srcPath = argv[1];

  auto [irOut, formatOut, patternOut] =
      getOutStreams(InputFilename, OutputFilename, true);

  TokenStream Ts(InputFilename.c_str());
  LLVMContext Ctx;
  auto Mod = std::make_unique<Module>("mod", Ctx);
  auto Instrs = ParseCoreDSL2(Ts, (XLen == 64), Mod.get(), NoExtend);

  if (irOut) {
    std::string Str;
    raw_string_ostream OS(Str);
    OS << *Mod;
    OS.flush();
    irOut << Str << "\n";
    irOut.close();
  }

  if (!SkipVerify)
    if (verifyModule(*Mod, &errs()))
      return -1;

  if (PrintIR)
    llvm::outs() << *Mod << "\n";

  // TODO: use force

  llvm::CodeGenOptLevel Opt;
  switch (OptLevel) {
  case '0':
    Opt = llvm::CodeGenOptLevel::None;
    break;
  case '1':
    Opt = llvm::CodeGenOptLevel::Less;
    break;
  case '2':
    Opt = llvm::CodeGenOptLevel::Default;
    break;
  case '3':
    Opt = llvm::CodeGenOptLevel::Aggressive;
    break;
  }

  PGArgsStruct Args{// .Mattr = Mattr,
                    .OptLevel = Opt,
                    .Predicates = Predicates,
                    .Is64Bit = (XLen == 64)};

  optimizeBehavior(Mod.get(), Instrs, irOut, Args);
  if (PrintIR)
    llvm::outs() << *Mod << "\n";
  if (!SkipFmt)
    PrintInstrsAsTableGen(Instrs, formatOut);

  if (!SkipPat)
    if (generatePatterns(Mod.get(), Instrs, patternOut, Args))
      return -1;
  return 0;
}
