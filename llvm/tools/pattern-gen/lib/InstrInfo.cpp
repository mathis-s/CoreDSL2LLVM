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
        if (f.type & (CDSLInstr::FieldType::NON_CONST))
        {
            int len = f.len;
            s << "\tbits<" << len << "> " << f.ident << ";\n";
        }

    for (auto const& f : instr.frags)
    {
        s << "\tlet Inst{" << std::to_string(f.dstOffset + f.len - 1) << "-" << std::to_string(f.dstOffset) << "} = ";
        auto const& field = instr.fields[f.idx];

        if (field.type == CDSLInstr::FieldType::CONST)
            s << "0x" << std::hex << ((field.constV >> f.srcOffset) & ((1UL << f.len) - 1)) << std::dec;
        else
            s << field.ident << "{" << std::to_string(f.srcOffset + f.len - 1) << "-"
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
