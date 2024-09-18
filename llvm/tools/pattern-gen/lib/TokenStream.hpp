#pragma once
#include <string>
#include <optional>
#include <map>
#include "Token.hpp"

struct TokenStream
{
    const std::string path;
    const std::string src;
    size_t i = 0;
    int lineNumber = 1;

  private:
    std::optional<Token> peekToken;
    std::map<std::string_view, uint32_t> strings =
    { // initialize with keywords
        std::make_pair("if",IfKeyword-TOK_KW_START),
        std::make_pair("else",ElseKeyword-TOK_KW_START),
        std::make_pair("while",WhileKeyword-TOK_KW_START),
        std::make_pair("for",ForKeyword-TOK_KW_START),
        std::make_pair("operands",OperandsKeyword-TOK_KW_START),
        std::make_pair("encoding",EncodingKeyword-TOK_KW_START),
        std::make_pair("assembly",AssemblyKeyword-TOK_KW_START),
        std::make_pair("behavior",BehaviorKeyword-TOK_KW_START),
        std::make_pair("extends",ExtendsKeyword-TOK_KW_START),
        std::make_pair("instructions",InstructionsKeyword-TOK_KW_START),
        std::make_pair("signed", SignedKeyword-TOK_KW_START),
        std::make_pair("unsigned", UnsignedKeyword-TOK_KW_START),
    };
    const size_t NUM_KEYWORDS = strings.size();

  public:
    TokenStream (std::string&& srcPath);
    Token Pop();
    Token Peek();
    unsigned GetIdentIdx(std::string_view ident);
    std::string_view GetIdent(unsigned identIdx);
};
