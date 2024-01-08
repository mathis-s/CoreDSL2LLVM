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

#include "lib/InstrInfo.hpp"
#include "lib/Parser.hpp"
#include "lib/TokenStream.hpp"
#include "lib/Token.hpp"
#include "lib/Lexer.hpp"
#include "PatternGen.hpp"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Support/InitLLVM.h"

static auto get_out_streams (std::string srcPath)
{
    std::string outPath{srcPath};
    const char type[] = ".core_desc";
    if (outPath.find(type, outPath.size() - sizeof(type)))
        outPath = outPath.substr(0, outPath.size() - sizeof(type) + 1);
    return std::make_tuple(std::ofstream(outPath + "InstrFormat.td"), std::ofstream(outPath + ".td"));
}

int main (int argc, char** argv)
{
    if (argc <= 1)
    {
        fprintf(stderr, "usage: %s <SOURCE FILE> LLVM-ARGS...\n", argv[0]);
        return -1;
    }

    const char* srcPath = argv[1];

    auto [formatOut, patternOut] = get_out_streams(srcPath);

    TokenStream ts(srcPath);
    llvm::LLVMContext ctx;
    auto mod = std::make_unique<llvm::Module>("mod", ctx);
    auto instrs = ParseCoreDSL2(ts, mod.get());

    PrintInstrsAsTableGen(instrs, formatOut);

    llvm::outs() << *mod << "\n";
    if (llvm::verifyModule(*mod, &llvm::errs()))
        return -1;
    
    std::string extName = (argc > 2) ? argv[2] : "Xcvsimd"; 

    GeneratePatterns(mod.get(), instrs, patternOut, extName);
}
