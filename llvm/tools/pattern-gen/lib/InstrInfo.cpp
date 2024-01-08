#include "InstrInfo.hpp"
#include <sstream>
#include <algorithm>

std::string EncodingToTablgen(CDSLInstr const& instr)
{
    std::stringstream s;
    std::string opcodeString = instr.name;
    std::replace(opcodeString.begin(), opcodeString.end(), '_', '.');
    std::transform(opcodeString.begin(), opcodeString.end(), opcodeString.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    s << "class RVInst_" << instr.name << "<dag outs, dag ins>"
      << " : RVInst<outs, ins, \"" << opcodeString << "\", \"" << instr.argString << "\", [], InstFormatOther> {\n";

    for (auto const& f : instr.fields)
        if (f.IsRegImm() && f.srcOffset == 0)
        {
            int len = (f.type >= CDSLInstr::Field::ID_IMM0) ? f.value & 0x7fffffff : 5;
            s << "\tbits<" << len << "> " << f.TypeAsString() << ";\n";
        }

    for (auto const& f : instr.fields)
    {
        if (f.dstOffset == 0 && f.len == 7)
            s << "\tlet Opcode = ";
        else
            s << "\tlet Inst{" << std::to_string(f.dstOffset + f.len - 1) << "-" << std::to_string(f.dstOffset)
              << "} = ";
        if (f.type == CDSLInstr::Field::CONST)
            s << "0x" << std::hex << f.value << std::dec;
        else
            s << f.TypeAsString() << "{" << std::to_string(f.srcOffset + f.len - 1) << "-"
              << std::to_string(f.srcOffset) << "}";
        s << ";\n";
    }
    s << "}";
    return s.str();
}

void PrintInstrsAsTableGen (std::vector<CDSLInstr> const& instrs, std::ostream& ostream)
{
    for (auto const& instr : instrs)
        ostream << EncodingToTablgen(instr) << '\n';
}
