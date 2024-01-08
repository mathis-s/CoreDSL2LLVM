#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <array>
#include <cassert>

struct CDSLInstr
{
    struct Field
    {
        enum FieldType
        {
            CONST,
            ID_RD,
            ID_RS1,
            ID_RS2,
            ID_IMM0,
            ID_IMM1,
        };

        FieldType type;
        uint8_t len;
        uint8_t dstOffset;
        uint8_t srcOffset;
        uint32_t value; // for IMM fields
                        // sign bit: 1 -> signed, 0 -> unsigned
                        // remaining 31 bits: length of entire immediate (might differ from len)

        static inline Field ConstField(uint32_t value, uint8_t offset, uint8_t len)
        {
            return (Field)
            {
                .type = CONST,
                .len = len,
                .dstOffset = offset,
                .srcOffset = 0,
                .value = value
            };
        }

        static inline Field RegImmField(int reg, uint8_t dstOffset, uint8_t srcOffset, uint8_t len, bool immS = false, uint32_t immSize = 0)
        {
            return (Field)
            {
                .type = static_cast<FieldType>(ID_RD + reg),
                .len = len,
                .dstOffset = dstOffset,
                .srcOffset = srcOffset,
                .value = ((uint32_t)immS << 31) | (immSize & 0x7fffffff)
            };
        }

        bool inline IsRegImm() const
        {
            return type >= ID_RD && type <= ID_IMM1;
        }
        
        const inline std::string TypeAsString() const
        {
            // both signed and unsigned immediates are "imm"
            static const std::array<std::string, 6> fieldNames = 
            {
                "const", "rd", "rs1", "rs2", "imm", "imm2"
            };
            return fieldNames[type];
        }
    };

    inline bool SignedImm(int idx) const
    {
        for (auto const& field : fields)
        {
            if (field.type == Field::ID_IMM0 + idx)
                return (field.value & (1 << 31));
        }
        return false;
    }

    inline int GetImmLen(int idx) const
    {
        for (auto const& field : fields)
        {
            if (field.type == Field::ID_IMM0 + idx)
                return (field.value & 0x7fffffff);
        }
        return -1;
    }
    
    inline bool SetSignedImm(int idx, bool isSigned)
    {
        bool rv = false;
        for (auto& field : fields)
            if (field.type == Field::ID_IMM0 + idx)
            {
                field.value = (field.value & 0x7ffffffff) | (isSigned << 31);
                rv = true;
            }
        return rv;
    }

    std::string name;
    std::string argString;
    std::vector<Field> fields;
};

std::string EncodingToTablgen(CDSLInstr const& instr);
void PrintInstrsAsTableGen (std::vector<CDSLInstr> const& instrs, std::ostream& ostream);
