#pragma once
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>
#include <array>
#include <cassert>

struct CDSLInstr
{
    struct FieldFrag
    {
        uint8_t idx; // index into Field array
        uint8_t len;
        uint8_t dstOffset;
        uint8_t srcOffset;
    };

    enum FieldType
    {
        CONST = 0,
        NON_CONST = 1,
        SIGNED = 2,
        SIGNED_REG = 4,
        IMM = 8,
        REG = 16,
        IN = 32,
        OUT = 64,
    };

    struct Field
    {
        uint8_t len;
        uint32_t constV;
        std::string_view ident;
        uint32_t identIdx;
        FieldType type;
    };

    std::string name;
    std::string argString;

    llvm::SmallVector<Field, 4> fields;
    llvm::SmallVector<FieldFrag, 8> frags;
};

std::string EncodingToTablgen(CDSLInstr const& instr);
void PrintInstrsAsTableGen (std::vector<CDSLInstr> const& instrs, std::ostream& ostream);
