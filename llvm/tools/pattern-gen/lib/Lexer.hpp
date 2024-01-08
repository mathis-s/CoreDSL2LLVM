#pragma once
#include <cstddef>
#include "Token.hpp"

inline size_t LexOperator (const char* code, size_t i, TokenType* token)
{
switch (code[i + 0])
{
    case '!':
        if (code[i + 1] == '=')
        {
            *token = NotEquals;
            return i + 2;
        }
        *token = LogicalNOT;
        return i + 1;
    case '%':
        if (code[i + 1] == '=')
        {
            *token = AssignmentMod;
            return i + 2;
        }
        *token = Modulo;
        return i + 1;
    case '&':
        switch (code[i + 1])
        {
            case '&': *token = LogicalAND; return i + 2;
            case '=': *token = AssignmentAND; return i + 2;
        }
        *token = BitwiseAND;
        return i + 1;
    case '(': *token = RBrOpen; return i + 1;
    case ')': *token = RBrClose; return i + 1;
    case '*':
        if (code[i + 1] == '=')
        {
            *token = AssignmentMul;
            return i + 2;
        }
        *token = Multiply;
        return i + 1;
    case '+':
        switch (code[i + 1])
        {
            case '+': *token = Increment; return i + 2;
            case '=': *token = AssignmentAdd; return i + 2;
            case ':': *token = PlusColon; return i + 2;
        }
        *token = Plus;
        return i + 1;
    case ',': *token = Comma; return i + 1;
    case '-':
        switch (code[i + 1])
        {
            case '-': *token = Decrement; return i + 2;
            case '=': *token = AssignmentSub; return i + 2;
            //case '>': *token = Arrow; return i + 2;
            case ':': *token = MinusColon; return i + 2;
        }
        *token = Minus;
        return i + 1;
    //case '.': *token = Dot; return i + 1;
    case '/':
        if (code[i + 1] == '=')
        {
            *token = AssignmentDiv;
            return i + 2;
        }
        *token = Divide;
        return i + 1;
    case ':':
        if (code[i + 1] == ':')
        {
            *token = BitwiseConcat;
            return i + 2;
        }
        *token = Colon;
        return i + 1;
    case ';': *token = Semicolon; return i + 1;
    case '<':
        switch (code[i + 1])
        {
            case '<':
                if (code[i + 2] == '=')
                {
                    *token = AssignmentShiftLeft;
                    return i + 3;
                }
                *token = ShiftLeft;
                return i + 2;
            case '=': *token = LessThanEq; return i + 2;
        }
        *token = LessThan;
        return i + 1;
    case '=':
        if (code[i + 1] == '=')
        {
            *token = Equals;
            return i + 2;
        }
        *token = Assignment;
        return i + 1;
    case '>':
        switch (code[i + 1])
        {
            case '=': *token = GreaterThanEq; return i + 2;
            case '>':
                if (code[i + 2] == '=')
                {
                    *token = AssignmentShiftRight;
                    return i + 3;
                }
                *token = ShiftRight;
                return i + 2;
        }
        *token = GreaterThan;
        return i + 1;
    case '?': *token = Ternary; return i + 1;
    case '[': *token = ABrOpen; return i + 1;
    case ']': *token = ABrClose; return i + 1;
    case '^':
        if (code[i + 1] == '=')
        {
            *token = AssignmentXOR;
            return i + 2;
        }
        *token = BitwiseXOR;
        return i + 1;
    case '{': *token = CBrOpen; return i + 1;
    case '|':
        switch (code[i + 1])
        {
            case '=': *token = AssignmentOR; return i + 2;
            case '|': *token = LogicalOR; return i + 2;
        }
        *token = BitwiseOR;
        return i + 1;
    case '}': *token = CBrClose; return i + 1;
    case '~': *token = BitwiseNOT; return i + 1;
}
return 0;
}
