#include <exception>

#include <llvm/IR/LLVMContext.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <map>
#include <ctype.h>
#include <cstdio>
#include <fstream>
#include <tuple>
#include <memory>
#include <filesystem>

#include "lib/InstrInfo.hpp"
#include "lib/Parser.hpp"
#include "lib/TokenStream.hpp"
#include "lib/Token.hpp"
#include "lib/Lexer.hpp"
#include "PatternGen.hpp"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;


static cl::OptionCategory ToolOptions("Tool Options");
static cl::OptionCategory ViewOptions("View Options");

static cl::opt<std::string> InputFilename(cl::Positional,
                                          cl::desc("<input file>"),
                                          cl::cat(ToolOptions), cl::init("-"));

static cl::opt<std::string> OutputFilename("o", cl::desc("Output filename"),
                                           cl::init("-"), cl::cat(ToolOptions),
                                           cl::value_desc("filename"));

static cl::opt<std::string> InputLanguage("x", cl::desc("Input language ('cdsl' or 'll')"), cl::cat(ToolOptions));

static cl::opt<bool> Force ("f", cl::desc("Ignore parser errors."), cl::cat(ToolOptions));
static cl::opt<bool> Skip ("s", cl::desc("Skip pattern-gen step."), cl::cat(ToolOptions));

static cl::opt<std::string> ExtName("ext", cl::desc("Target extension"), cl::cat(ToolOptions), cl::init("Xcvsimd"));
static cl::opt<std::string> Mattr("mattr2", cl::desc("Target specific attributes"), cl::value_desc("a1,+a2,-a3,..."), cl::cat(ToolOptions), cl::init("+m,+unaligned-scalar-mem,+xcvalu,+xcvsimd"));

// Determine optimization level.
static cl::opt<char>
    OptLevel("O",
             cl::desc("Optimization level. [-O0, -O1, -O2, or -O3] "
                      "(default = '-O2')"),
             cl::cat(ToolOptions), cl::init('3'));

#include <iostream>
namespace fs = std::filesystem;

static auto get_out_streams (std::string srcPath, std::string destPath)
{
    fs::path outPath{destPath};
    std::string ext = outPath.extension();
    std::string stem = outPath.stem();
    // std::string name = outPath.filename();
    fs::path parent = outPath.parent_path();
    // std::cout << "outPath=" << outPath << "!" << std::endl;
    // std::cout << "ext=" <<  ext << "!" << std::endl;
    // std::cout << "stem=" <<  stem << "!" << std::endl;
    // std::cout << "name=" <<  name << "!" << std::endl;
    // std::cout << "parent=" <<  parent << "!" << std::endl;
    fs::path inPath{srcPath};
    std::string ext2 = inPath.extension();
    std::string stem2 = inPath.stem();
    // std::string name2 = inPath.filename();
    fs::path parent2 = inPath.parent_path();
    // std::cout << "inPath2=" << inPath << "!" << std::endl;
    // std::cout << "ext2=" <<  ext2 << "!" << std::endl;
    // std::cout << "stem2=" <<  stem2 << "!" << std::endl;
    // std::cout << "name2=" <<  name2 << "!" << std::endl;
    // std::cout << "parent2=" <<  parent2 << "!" << std::endl;
    fs::path basePath = parent2 / stem2;
    std::string newExt = ".td";
    if (outPath.compare("-") != 0) {
        basePath = parent / stem;
        newExt = ext;
    }
    // TODO: allow .td in out path
    std::string irPath = basePath.string() + ".ll";
    std::string fmtPath = basePath.string() + "InstrFormat" + newExt;
    std::string patPath = basePath.string() + newExt;
    // std::cout << "basePath=" << basePath << "!" << std::endl;
    // std::cout << "newExt=" << newExt << "!" << std::endl;
    // std::cout << "irPath=" << irPath << "!" << std::endl;
    // std::cout << "fmtPath=" << fmtPath << "!" << std::endl;
    // std::cout << "patPath=" << patPath << "!" << std::endl;
    return std::make_tuple(std::ofstream(irPath), std::ofstream(fmtPath), std::ofstream(patPath));
}


int main (int argc, char** argv)
{
    cl::HideUnrelatedOptions({&ToolOptions, &ViewOptions});
    cl::ParseCommandLineOptions(argc, argv, "CoreDSL2LLVM Pattern Gen");
    if (argc <= 1)
    {
        fprintf(stderr, "usage: %s <SOURCE FILE> LLVM-ARGS...\n", argv[0]);
        return -1;
    }

    // const char* srcPath = argv[1];

    auto [irOut, formatOut, patternOut] = get_out_streams(InputFilename, OutputFilename);

    TokenStream ts(InputFilename.c_str());
    LLVMContext ctx;
    auto mod = std::make_unique<Module>("mod", ctx);
    auto instrs = ParseCoreDSL2(ts, mod.get());


    if (irOut) {
        // outs() << *mod << "\n";
        std::string Str;
        raw_string_ostream OS(Str);
        OS << *mod;
        OS.flush();
        irOut << Str << "\n";
        irOut.close();
    }
    if (verifyModule(*mod, &errs()))
        return -1;

    // TODO: use opt_level
    // TODO: use force
    // TODO: write ll to file
    // TODO: print optimized llvm ir

    if (!Skip) {
        PrintInstrsAsTableGen(instrs, formatOut);
        // std::string extName = (argc > 2) ? argv[2] : "Xcvsimd";

        GeneratePatterns(mod.get(), instrs, patternOut, ExtName, OptLevel, Mattr);
    }
}
