#pragma once
#include <stdint.h>
#include <memory>
#include <string_view>

enum TokenType
{
    // These tokens can be associated with
    // binops or postfix unops (or ternary).
    // They should be in one block for precedence LUT.
    OP_START = 0,
    ABrOpen = OP_START,
    Plus,
    Minus,
    Multiply,
    Divide,
    Modulo,
    ShiftLeft,
    ShiftRight,
    LessThan,
    GreaterThan,
    LessThanEq,
    GreaterThanEq,
    Equals,
    NotEquals,
    BitwiseAND,
    BitwiseOR,
    BitwiseXOR,
    LogicalAND,
    LogicalOR,
    Ternary,
    BitwiseConcat,

    Assignment,
    AssignmentAdd,
    AssignmentSub,
    AssignmentMul,
    AssignmentDiv,
    AssignmentMod,
    AssignmentAND,
    AssignmentOR,
    AssignmentXOR,
    AssignmentShiftRight,
    AssignmentShiftLeft,
    Increment,
    Decrement,
    OP_END = Decrement,

    // Remaining tokens, order of these does not matter.
    None,
    RBrOpen,
    RBrClose,
    CBrOpen,
    CBrClose,
    ABrClose,
    BitwiseNOT,
    LogicalNOT,
    Colon,
    PlusColon,
    MinusColon,
    Comma,
    Semicolon,

    TOK_KW_START,
    IfKeyword = TOK_KW_START,
    ElseKeyword,
    WhileKeyword,
    ForKeyword,
    OperandsKeyword,
    EncodingKeyword,
    AssemblyKeyword,
    BehaviorKeyword,
    ExtendsKeyword,
    InstructionsKeyword,
    SignedKeyword,
    UnsignedKeyword,

    TOK_KW_END = UnsignedKeyword,

    Identifier,
    IntLiteral,
    StringLiteral,
};

struct Token
{
    TokenType type;
    union
    {
        struct
        {
            uint32_t idx;
            std::string_view str;
        } ident;
        struct
        {
            bool isSigned;
            unsigned bitLen;
            uint64_t value;
        } literal;
        struct
        {
            std::string_view str;
        } strLit;
    };

    Token (TokenType type) : type(type) {}
    Token (uint32_t identIdx, std::string_view&& identStr) : type(Identifier)
    {
        ident.str = identStr;
        ident.idx = identIdx;
    }
    Token (bool isSigned, unsigned bitLen, uint64_t value) : type(IntLiteral)
    {
        literal.isSigned = isSigned;
        literal.bitLen = bitLen;
        literal.value = value;
    }
    Token (std::string_view&& stringLiteral) : type(StringLiteral)
    {
        strLit.str = stringLiteral;
    }

    bool operator== (Token const& b)
    {
        if (b.type != type) return false;
        switch (type)
        {
            case Identifier: return b.ident.idx == ident.idx;
            case StringLiteral: return b.strLit.str == strLit.str;
            case IntLiteral: return
                b.literal.isSigned == literal.isSigned &&
                b.literal.bitLen == literal.bitLen &&
                b.literal.value == literal.value;
            default: return true;
        }
    }
    bool operator!= (Token const& b)
    {
        return !(*this == b);
    }

};
