#include "TokenStream.hpp"
#include "Lexer.hpp"
#include <iostream>
#include <fstream>
#include <cerrno>

static int clog2 (uint64_t n)
{
    if (n == 0) return 1;
    return std::max(64 - __builtin_clzl(n), 1);
}

Token TokenStream::Pop()
{
    if (peekToken.has_value())
    {
        Token t = peekToken.value();
        peekToken.reset();
        return t;
    }

    size_t len = src.length();
    const char* srcC = src.c_str();

    while (1)
    {
        // Skip whitespace
        while (isspace(srcC[i]))
        {
            if (srcC[i] == '\n') lineNumber++;
            i++;
        }

        // Single Line Comment
        if (srcC[i] == '/' && srcC[i + 1] == '/')
        {
            while (i < len && srcC[i] != '\n')
                i++;
            continue;
        }

        /* Multi-line
           Comment  */
        if (srcC[i] == '/' && srcC[i + 1] == '*')
        {
            i += 2;
            while (i < len && !(srcC[i - 2] == '*' && src[i - 1] == '/'))
            {
                if (srcC[i-2] == '\n') lineNumber++;
                i++;
            }
            continue;
        }
        break;
    }

    if (i == len)
        return Token(None);

    { // Try lexing operator
        TokenType t;
        size_t nextI = LexOperator(srcC, i, &t);
        if (nextI != 0)
        {
            i = nextI;
            return Token(t);
        }
    }

    // Try lexing keywords or tokens
    if (isalpha(srcC[i]) || srcC[i] == '_')
    {
        size_t identLen = 1;
        while (1)
        {
            char c = srcC[i + identLen];
            if (!isdigit(c) && !isalpha(c) && c != '_')
                break;
            identLen++;
        }
        std::string_view substr(srcC + i, identLen);
        i += identLen;
        auto iter = strings.find(substr);
        if (iter != strings.end())
        {
            if (iter->second < NUM_KEYWORDS)
                return Token((TokenType)(TOK_KW_START + iter->second));
            else
                return Token(iter->second - NUM_KEYWORDS, std::move(substr));
        }
        else
        {
            uint32_t idx = (strings[substr] = strings.size());
            return Token(idx - NUM_KEYWORDS, std::move(substr));
        }
    }

    // Int Literal
    if (isdigit(srcC[i]))
    {
        bool isBase10 = srcC[i] != 0 && srcC[i + 1] != 'b' && srcC[i + 1] != 'x';
        uint64_t literal;
        size_t iCopy = i;

        {
            const char* startPtr = src.c_str() + iCopy;
            char* endPtr;
            literal = strtoul(startPtr, &endPtr, 0);
            if (startPtr == endPtr) return Token(None);
            iCopy += endPtr - startPtr;
        }

        if (isBase10 && srcC[iCopy] == '\'')
        {
            // Do not support sized literals larger than 64 bit
            if (literal > 64)
                return Token(None);

            iCopy++;
            bool isSigned = (srcC[iCopy] == 's');
            if (isSigned)
                iCopy++;

            int base;
            switch (srcC[iCopy])
            {
                case 'h': base = 16; break;
                case 'd': base = 10; break;
                case 'o': base = 8; break;
                case 'b': base = 2; break;
                default: return Token(None);
            }
            iCopy++;

            size_t literal2;
            {
                const char* startPtr = src.c_str() + iCopy;
                char* endPtr;
                literal2 = strtoul(startPtr, &endPtr, base);
                if (startPtr == endPtr) return Token(None);
                iCopy += endPtr - startPtr;
            }


            i = iCopy;
            return Token(isSigned, literal, literal2);
        }
        i = iCopy;
        return Token(false, clog2(literal), literal);
    }

    // String Literal
    if (srcC[i] == '\"')
    {
        size_t litLen = 1;
        if (i + litLen >= len)
            return Token(None);
        while (srcC[i + litLen] != '\"')
        {
            if (srcC[i + litLen] == '\n')
                return Token(None);
            litLen++;
            if (i + litLen >= len)
                return Token(None);
        }

        Token t = Token(std::string_view(srcC + i + 1, litLen - 1));
        i += litLen + 1;
        return t;
    }

    return Token(None);
}

Token TokenStream::Peek()
{
    if (!peekToken.has_value())
        peekToken = Pop();
    return peekToken.value();
}

static std::string read_file_as_str (std::string path)
{
    std::ifstream ifs(path);
    if (!ifs) {
        fprintf(stderr, "Aborting! File does not exist: %s\n", path.c_str());
        exit(-1);
    }
    std::string content((std::istreambuf_iterator<char>(ifs)),
                        std::istreambuf_iterator<char>());

    return content;
}

TokenStream::TokenStream(std::string&& srcPath) : path(srcPath), src(read_file_as_str(srcPath)) {}
