#include <exception>

#include <cstdio>
#include <ctype.h>
#include <filesystem>
#include <fstream>
#include <llvm/IR/LLVMContext.h>
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
static cl::opt<bool> SkipFmt("skip-formats", cl::desc("Skip tablegen formats step."),
                          cl::cat(ToolOptions));
static cl::opt<bool> SkipPat("skip-patterns", cl::desc("Skip pattern-gen step."),
                          cl::cat(ToolOptions));
static cl::opt<bool> SkipVerify("skip-verify", cl::desc("Skip verification step."),
                          cl::cat(ToolOptions));
static cl::opt<bool> PrintIR("print-ir", cl::desc("Print LLVM-IR module."),
                          cl::cat(ToolOptions));

static cl::opt<std::string> ExtName("ext", cl::desc("Target extension"),
                                    cl::cat(ToolOptions), cl::init("ExtXcvsimd"));
static cl::opt<std::string>
    Mattr("mattr2", cl::desc("Target specific attributes"),
          cl::value_desc("a1,+a2,-a3,..."), cl::cat(ToolOptions),
          // cl::init("+m,+fast-unaligned-access,+xcvalu,+xcvsimd"));
          cl::init("+m,+fast-unaligned-access"));

// Determine optimization level.
static cl::opt<char>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O3')"),
             cl::cat(ToolOptions), cl::init('3'));

static cl::opt<std::string> Predicates(
    "p", cl::desc("Predicate(s) used for instructions in output TableGen"), cl::cat(ToolOptions), cl::init("HasVendorXCValu"));

#include <iostream>
namespace fs = std::filesystem;

static auto get_out_streams(std::string srcPath, std::string destPath, bool emitLL) {
  fs::path outPath{destPath};

  fs::path inPath{srcPath};
  fs::path basePath = inPath.parent_path() / inPath.stem();

  std::string newExt = ".td";
  if (outPath.compare("-") != 0) {
    basePath = outPath.parent_path() / outPath.stem();
    newExt = outPath.extension();
  }
  // TODO: allow .td in out path
  std::string irPath = "/dev/null";
  std::string fmtPath = "/dev/null";
  std::string patPath = "/dev/null";
  if (emitLL) {
    irPath = basePath.string() + ".ll";
  }
  if (!SkipFmt) {
    fmtPath = basePath.string() + "InstrFormat" + newExt;
  }
  if (!SkipPat) {
    patPath = basePath.string() + newExt;
  }

  return std::make_tuple(std::ofstream(irPath), std::ofstream(fmtPath),
                         std::ofstream(patPath));
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
      get_out_streams(InputFilename, OutputFilename, true);

  TokenStream ts(InputFilename.c_str());
  LLVMContext ctx;
  auto mod = std::make_unique<Module>("mod", ctx);
  auto instrs = ParseCoreDSL2(ts, mod.get());

  if (!SkipVerify)
    if (verifyModule(*mod, &errs()))
      return -1;

  if (PrintIR)
    llvm::outs() << *mod << "\n";

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

  PGArgsStruct Args{.ExtName = ExtName,
                    .Mattr = Mattr,
                    .OptLevel = Opt,
                    .Predicates = Predicates};

  OptimizeBehavior(mod.get(), instrs, irOut, Args);
  if (PrintIR)
    llvm::outs() << *mod << "\n";
  if (!SkipFmt)
    PrintInstrsAsTableGen(instrs, formatOut);

  if (!SkipPat)
    if (GeneratePatterns(mod.get(), instrs, patternOut, Args))
      return -1;
  return 0;
}
