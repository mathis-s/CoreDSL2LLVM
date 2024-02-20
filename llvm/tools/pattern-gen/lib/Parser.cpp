#include "Parser.hpp"
#include "Token.hpp"
#include "TokenStream.hpp"
#include "InstrInfo.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/TypeSize.h"
#include <array>
#include <cstdlib>
#include <functional>
#include <regex>
#include <sstream>
#include <string>
#include <unordered_map>

const size_t IMM_CNT = 2;
using namespace std::placeholders;

struct Value
{
    llvm::Value* ll;
    int bitWidth;
    bool isSigned;
    bool isLValue;

    Value(llvm::Value* llvalue, bool isSigned = false) : ll(llvalue), isSigned(isSigned)
    {
        assert(!llvm::isa<llvm::PointerType>(llvalue->getType()));
        bitWidth = llvalue->getType()->getIntegerBitWidth();
        isLValue = false;
    }

    Value(llvm::Value* llvalue, int bitWidth, bool isSigned = false)
        : ll(llvalue), bitWidth(bitWidth), isSigned(isSigned)
    {
        isLValue = true;
    }

    Value()
    {
    }
};
struct Variable
{
    Value val;
    int scope;
};
static llvm::DenseMap<uint32_t, llvm::SmallVector<Variable>> variables;
static llvm::DenseMap<llvm::StringRef, int> immediateNames;
static int scopeDepth;
static llvm::BasicBlock* entry;
static CDSLInstr* curInstr;
static void reset_globals()
{
    variables.clear();
    immediateNames.clear();
    scopeDepth = 0;
    entry = nullptr;
    curInstr = nullptr;
}

static void create_var_scope()
{
    scopeDepth++;
}

static void pop_var_scope()
{
    for (auto it = variables.begin(); it != variables.end();)
    {
        auto& [vname, vstack] = *it++;
        assert(!vstack.empty());
        if (vstack.back().scope == scopeDepth)
            vstack.pop_back();
        if (vstack.empty())
            variables.erase(vname);
    }
    
    scopeDepth--;
}

void ParseScope(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build);
Value ParseExpression(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, int minPrec = 0);

static void __attribute__((noreturn)) error(const char* msg, TokenStream& ts)
{
    fprintf(stderr, "%s:%i: %s\n", ts.path.c_str(), ts.lineNumber, msg);
    exit(-1);
}

static void __attribute__((noreturn)) syntax_error(TokenStream& ts)
{
    error("syntax error", ts);
}

static void __attribute__((noreturn)) not_implemented(TokenStream& ts)
{
    error("not implemented", ts);
}

static Token pop_cur(TokenStream& ts, TokenType expected)
{
    Token t = ts.Pop();
    if (t.type != expected)
        syntax_error(ts);
    return t;
}

static bool pop_cur_if(TokenStream& ts, TokenType expected)
{
    Token t = ts.Peek();
    if (t.type == expected)
    {
        ts.Pop();
        return true;
    }
    return false;
}

static int ceil_to_pow2 (int n)
{
    int bitWidth8 = (((n + 7) / 8) * 8);
    int bitWidth2 = 1 << (31 - __builtin_clz(bitWidth8));
    if (bitWidth8 != bitWidth2)
        bitWidth2 *= 2;
    return bitWidth2;
}

void promote_lvalue(llvm::IRBuilder<>& build, Value& v)
{
    if (!v.isLValue)
        return;
    v.ll = build.CreateLoad(llvm::Type::getIntNTy(build.getContext(), v.bitWidth), v.ll,
        v.ll->getName() + ".v");
    v.isLValue = false;
}

static void fit_to_size(Value& v, llvm::IRBuilder<>& build)
{
    bool sExt = v.isSigned;
    auto& ctx = build.getContext();
    // To give LLVM an easier time, generate
    // expressions with power-of-two bit widths
    
    int bitWidth2 = ceil_to_pow2(v.bitWidth);
    if (bitWidth2 > 32) bitWidth2 = 32;
    llvm::Type* newType = llvm::Type::getIntNTy(ctx, bitWidth2);

    if (newType->getIntegerBitWidth() > (v.isLValue ? v.bitWidth : v.ll->getType()->getIntegerBitWidth()))
        v.ll = sExt ? build.CreateSExt(v.ll, newType) : build.CreateZExt(v.ll, newType);
    else if (newType->getIntegerBitWidth() < (v.isLValue ? v.bitWidth : v.ll->getType()->getIntegerBitWidth()))
        v.ll = build.CreateTrunc(v.ll, newType);
    //v.bitWidth = bitWidth2;
}

Value gen_subscript(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left,
                    Value right)
{
    auto& ctx = func->getContext();
    Value upper = ParseExpression(ts, func, build);
    
    int len = 0;
    Value lower;
    
    if (pop_cur_if(ts, Colon))
    {
        lower = ParseExpression(ts, func, build);
        if (!llvm::isa<llvm::ConstantInt>(lower.ll) || !llvm::isa<llvm::ConstantInt>(upper.ll))
            not_implemented(ts);
        len = llvm::cast<llvm::ConstantInt>(upper.ll)->getLimitedValue() - 
            llvm::cast<llvm::ConstantInt>(lower.ll)->getLimitedValue() + 1;
    }
    else if (pop_cur_if(ts, PlusColon))
    {
        len = pop_cur(ts, IntLiteral).literal.value;
        lower = upper;
        upper = build.CreateAdd(upper.ll, llvm::ConstantInt::get(upper.ll->getType(), len));
    }
    else if (pop_cur_if(ts, MinusColon))
    {
        len = pop_cur(ts, IntLiteral).literal.value;
        lower = build.CreateAdd(upper.ll, llvm::ConstantInt::get(upper.ll->getType(), len));
    }
    else
    {
        lower = upper;
        len = 1;
    }
    pop_cur(ts, ABrClose);
    
    for (auto& v : {&lower, &upper})
    {
        promote_lvalue(build, *v);
        if (v->bitWidth != 32)
        {
            v->isSigned = false;
            v->bitWidth = 32;
            fit_to_size(lower, build);
        }
    }
    
    if (left.isLValue && (len == 8 || len == 16 || len == 32) && left.bitWidth == 32)
    {
        if (auto asConst = llvm::dyn_cast<llvm::ConstantInt>(lower.ll))
            if (asConst->getLimitedValue() % 8 != 0)
                not_implemented(ts);

        auto offset = build.CreateUDiv(lower.ll, llvm::ConstantInt::get(lower.ll->getType(), len));
        offset = build.CreateAnd(offset, llvm::ConstantInt::get(lower.ll->getType(), 3));
        
        auto llptr = build.CreateGEP(llvm::Type::getIntNTy(ctx, len), left.ll,
                                     {offset});

        left.ll = llptr;
        left.bitWidth = len;
        left.isLValue = true;
    }
    else if (left.bitWidth == 32 && (len == 8 || len == 16))
    {
        promote_lvalue(build, left);

        auto ec = llvm::ElementCount::getFixed(32 / len);
        left.ll = build.CreateBitCast(left.ll, llvm::VectorType::get(llvm::Type::getIntNTy(ctx, len), ec));
        
        upper.bitWidth = left.bitWidth;
        fit_to_size(upper, build);
        auto* idx = build.CreateUDiv(upper.ll, llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), len));

        left.ll = build.CreateExtractElement(left.ll, idx);
        left.bitWidth = len;
        fit_to_size(left, build);
    }
    else
    {
        promote_lvalue(build, left);

        lower.bitWidth = left.bitWidth;
        fit_to_size(lower, build);
        upper.bitWidth = left.bitWidth;
        fit_to_size(upper, build);

        left.ll = build.CreateLShr(left.ll, lower.ll);
        int llLen = upper.ll->getType()->getIntegerBitWidth();
        llvm::Value* mask = (len == llLen) ? llvm::ConstantInt::get(upper.ll->getType(), 0) : 
            build.CreateShl(llvm::ConstantInt::get(upper.ll->getType(), 1), len);
        mask = build.CreateSub(mask, llvm::ConstantInt::get(upper.ll->getType(), 1));
        left.ll = build.CreateAnd(left.ll, mask);
        
        left.bitWidth = len;
        fit_to_size(left, build);
    }
    return left;
}

Value gen_assign(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left, Value right)
{
    auto& ctx = func->getContext();

    if (!left.isLValue)
        error("cannot assign rvalue", ts);

    if (left.bitWidth < right.bitWidth && op == Assignment)
        error("implicit truncation", ts);

    Value rightOriginal = right;

    promote_lvalue(build, right);
    right.bitWidth = left.bitWidth;
    fit_to_size(right, build);

    build.CreateStore(right.ll, left.ll);

    return rightOriginal;
}

Value gen_compare(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left,
                  Value right)
{
    using Op = llvm::CmpInst::Predicate;

    promote_lvalue(build, left);
    promote_lvalue(build, right);

    static const Op conv[] = {
        [ABrOpen] = Op::BAD_ICMP_PREDICATE,
        [Plus] = Op::BAD_ICMP_PREDICATE,
        [Minus] = Op::BAD_ICMP_PREDICATE,
        [Multiply] = Op::BAD_ICMP_PREDICATE,
        [Divide] = Op::BAD_ICMP_PREDICATE,
        [Modulo] = Op::BAD_ICMP_PREDICATE,
        [ShiftLeft] = Op::BAD_ICMP_PREDICATE,
        [ShiftRight] = Op::BAD_ICMP_PREDICATE,
        [LessThan] = Op::ICMP_SLT,
        [GreaterThan] = Op::ICMP_SGT,
        [LessThanEq] = Op::ICMP_SLE,
        [GreaterThanEq] = Op::ICMP_SGE,
        [Equals] = Op::ICMP_EQ,
        [NotEquals] = Op::ICMP_NE,
    };
    Op llop = conv[op];

    // CoreDSL2 does not specify explictly under what circumstances ordering
    // comparisons are signed or unsigned. Assume usual C behavior.
    // "[Comparison operators] do not take the operand types into account"
    // -> doesn't make sense
    bool signedCmp = left.isSigned || right.isSigned;
    if (!signedCmp)
        switch (llop)
        {
            case Op::ICMP_SGE: llop = Op::ICMP_UGE; break;
            case Op::ICMP_SLE: llop = Op::ICMP_ULE; break;
            case Op::ICMP_SGT: llop = Op::ICMP_UGT; break;
            case Op::ICMP_SLT: llop = Op::ICMP_ULT; break;
            default: break;
        }

    int compareWidth = std::max(left.bitWidth, right.bitWidth);
    left.bitWidth = compareWidth;
    fit_to_size(left, build);
    right.bitWidth = compareWidth;
    fit_to_size(right, build);

    Value result;
    result.isLValue = false;
    result.isSigned = false;
    result.bitWidth = 1;
    result.ll = build.CreateICmp(llop, left.ll, right.ll);
    return result;
}

Value gen_binop(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left, Value right)
{
    using Op = llvm::BinaryOperator;
    auto& ctx = func->getContext();

    Value leftOriginal = left;
    promote_lvalue(build, left);
    promote_lvalue(build, right);

    static const Op::BinaryOps conv[] = {
        [ABrOpen] = Op::BinaryOps::BinaryOpsEnd,
        [Plus] = Op::Add,
        [Minus] = Op::Sub,
        [Multiply] = Op::Mul,
        [Divide] = Op::SDiv,
        [Modulo] = Op::SRem,
        [ShiftLeft] = Op::Shl,
        [ShiftRight] = Op::AShr,
        [LessThan] = Op::BinaryOpsEnd,
        [GreaterThan] = Op::BinaryOpsEnd,
        [LessThanEq] = Op::BinaryOpsEnd,
        [GreaterThanEq] = Op::BinaryOpsEnd,
        [Equals] = Op::BinaryOpsEnd,
        [NotEquals] = Op::BinaryOpsEnd,
        [BitwiseAND] = Op::And,
        [BitwiseOR] = Op::Or,
        [BitwiseXOR] = Op::Xor,
        [LogicalAND] = Op::BinaryOpsEnd,
        [LogicalOR] = Op::BinaryOpsEnd,
        [Ternary] = Op::BinaryOpsEnd,
        [BitwiseConcat] = Op::BinaryOpsEnd,
        [Assignment] = Op::BinaryOpsEnd,
        [AssignmentAdd] = Op::Add,
        [AssignmentSub] = Op::Sub,
        [AssignmentMul] = Op::Mul,
        [AssignmentDiv] = Op::SDiv,
        [AssignmentMod] = Op::SRem,
        [AssignmentAND] = Op::And,
        [AssignmentOR] = Op::Or,
        [AssignmentXOR] = Op::Xor,
        [AssignmentShiftRight] = Op::Shl,
        [AssignmentShiftLeft] = Op::AShr,
    };
    assert(op >= Plus && op <= AssignmentShiftRight);
    llvm::BinaryOperator::BinaryOps llop = conv[op];

    bool outSigned;
    switch (llop)
    {
        case Op::Shl:
        case Op::AShr:
        case Op::SRem: outSigned = left.isSigned; break;
        case Op::Sub: outSigned = true; break;
        default: outSigned = left.isSigned || right.isSigned; break;
    }

    int w1 = left.bitWidth;
    int w2 = right.bitWidth;
    int signedPair = (left.isSigned << 1) | right.isSigned;

    int resultWidth;
    switch (llop)
    {
        case Op::Add:
            switch (signedPair)
            {
                case 0b00: resultWidth = std::max(w1, w2) + 1; break;
                case 0b11: resultWidth = std::max(w1, w2) + 1; break;
                case 0b10: resultWidth = std::max(w1, w2 + 1) + 1; break;
                case 0b01: resultWidth = std::max(w1 + 1, w2) + 1; break;
            }
            break;
        case Op::Sub:
            switch (signedPair)
            {
                case 0b00: resultWidth = std::max(w1 + 1, w2 + 1); break;
                case 0b11: resultWidth = std::max(w1 + 1, w2 + 1); break;
                case 0b10: resultWidth = std::max(w1, w2 + 1) + 1; break;
                case 0b01: resultWidth = std::max(w1 + 1, w2) + 1; break;
            }
            break;
        case Op::Mul: resultWidth = w1 + w2; break;
        case Op::SDiv: resultWidth = w1 + right.isSigned; break;
        case Op::SRem:
            switch (signedPair)
            {
                case 0b00: resultWidth = std::min(w1, w2); break;
                case 0b11: resultWidth = std::min(w1, w2); break;
                case 0b10: resultWidth = std::min(w1, w2 + 1); break;
                case 0b01: resultWidth = std::min(w1, std::max(1, w2 - 1)); break;
            }
            break;
        default: resultWidth = std::max(w1, w2);
    }

    if (!left.isSigned && !right.isSigned)
        switch (llop)
        {
            case Op::SDiv: llop = Op::UDiv; break;
            case Op::SRem: llop = Op::URem; break;
            case Op::AShr: llop = Op::LShr; break;
            default: break;
        }

    left.bitWidth = resultWidth;
    fit_to_size(left, build);

    right.bitWidth = resultWidth;
    fit_to_size(right, build);

    auto v = Value(build.CreateBinOp(llop, left.ll, right.ll), outSigned);

    if (op >= AssignmentAdd)
        return gen_assign(ts, func, build, op, leftOriginal, v);
    return v;
}

Value gen_ternary(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left,
                    Value right)
{
    (void)right;
    (void)op;
    auto& ctx = func->getContext();

    llvm::BasicBlock* blockTrue = llvm::BasicBlock::Create(ctx, "true", func);
    llvm::BasicBlock* blockFalse = llvm::BasicBlock::Create(ctx, "false", func);
    llvm::BasicBlock* blockTerm = llvm::BasicBlock::Create(ctx, "select", func);
    
    promote_lvalue(build, left);

    auto cond = build.CreateICmpNE(left.ll, llvm::ConstantInt::get(left.ll->getType(), 0));
    build.CreateCondBr(cond, blockTrue, blockFalse);
    
    build.SetInsertPoint(blockTrue);
    auto valTrue = ParseExpression(ts, func, build);
    promote_lvalue(build, valTrue);
    
    pop_cur(ts, Colon);

    build.SetInsertPoint(blockFalse);
    auto valFalse = ParseExpression(ts, func, build);
    promote_lvalue(build, valFalse);
    

    if (valTrue.ll->getType() != valFalse.ll->getType())
    {
        if (valTrue.bitWidth > valFalse.bitWidth)
        {
            build.SetInsertPoint(blockFalse);
            valFalse.bitWidth = valTrue.bitWidth;
            fit_to_size(valFalse, build);
        }
        else if (valTrue.bitWidth < valFalse.bitWidth)
        {
            build.SetInsertPoint(blockTrue);
            valTrue.bitWidth = valFalse.bitWidth;
            fit_to_size(valTrue, build);
        }
    }
    build.SetInsertPoint(blockFalse);
    build.CreateBr(blockTerm);
    build.SetInsertPoint(blockTrue);
    build.CreateBr(blockTerm);

    build.SetInsertPoint(blockTerm);
    
    auto retval = build.CreatePHI(valTrue.ll->getType(), 2);
    retval->addIncoming(valTrue.ll, blockTrue);
    retval->addIncoming(valFalse.ll, blockFalse);

    valTrue.ll = retval;
    return valTrue;
}

Value gen_logical(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left,
                  Value right)
{
    (void)right;
    const auto i1 = llvm::Type::getInt1Ty(func->getContext());

    auto& ctx = func->getContext();
    llvm::BasicBlock* blockPre = build.GetInsertBlock();
    llvm::BasicBlock* blockEvalRHS = llvm::BasicBlock::Create(ctx, "logEvalRHS", func);
    llvm::BasicBlock* blockPost = llvm::BasicBlock::Create(ctx, "logPost", func);

    llvm::BasicBlock* trueBl = blockEvalRHS;
    llvm::BasicBlock* falseBl = blockPost;
    if (op == LogicalOR)
        std::swap(trueBl, falseBl);
    promote_lvalue(build, left);
    build.CreateCondBr(build.CreateICmpNE(left.ll, llvm::ConstantInt::get(left.ll->getType(), 0)), trueBl, falseBl);

    build.SetInsertPoint(blockEvalRHS);

    // In the operator table this is handled as a unary op such that we can call ParseExpression
    // for RHS ourselves here (we may only evaluate RHS if LHS does not short circuit)
    auto valRight = ParseExpression(ts, func, build);
    promote_lvalue(build, valRight);
    auto valRightBool = build.CreateICmpNE(valRight.ll, llvm::ConstantInt::get(valRight.ll->getType(), 0));

    build.CreateBr(blockPost);
    blockEvalRHS = build.GetInsertBlock();
    
    build.SetInsertPoint(blockPost);
    
    auto phi = build.CreatePHI(i1, 2);
    phi->addIncoming(valRightBool, blockEvalRHS);
    phi->addIncoming(llvm::ConstantInt::get(i1, (op == LogicalOR) ? 1 : 0), blockPre);
    return Value(phi, false);
}

Value gen_concat(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left, Value right)
{
    promote_lvalue(build, left);
    promote_lvalue(build, right);

    left.bitWidth += right.bitWidth;
    fit_to_size(left, build);
    left.ll = build.CreateShl(left.ll, llvm::ConstantInt::get(left.ll->getType(), right.bitWidth));

    right.bitWidth = left.bitWidth;
    {
        bool rightSigned = right.isSigned;
        right.isSigned = false;
        fit_to_size(right, build);
        right.isSigned = rightSigned;
    }
    
    left.ll = build.CreateOr(left.ll, right.ll);
    return left;
}

Value gen_inc(bool isPost, TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left, Value right)
{
    auto& ctx = func->getContext();

    if (!left.isLValue)
        error("cannot assign rvalue", ts);
    
    auto pre = build.CreateLoad(llvm::Type::getIntNTy(ctx, left.bitWidth), left.ll);
    auto post = build.CreateAdd(pre, llvm::ConstantInt::get(pre->getType(), (op == Decrement) ? -1 : 1));
    build.CreateStore(post, left.ll);

    return Value(isPost ? post : pre, left.isSigned);
}

// this can be updated to std::bind_front once LLVM switches to a newer standard
auto gen_preinc = std::bind(&gen_inc, false, _1, _2, _3, _4, _5, _6);
auto gen_postinc = std::bind(&gen_inc, true, _1, _2, _3, _4, _5, _6);

struct Operator
{
    using OpSig = Value(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, TokenType op, Value left,
                         Value right);
    uint8_t prec;
    bool rassoc;
    bool unary;
    std::function<OpSig> func;
    Operator(uint8_t prec, bool rassoc, bool unary, std::function<OpSig>&& func)
        : prec(15 - prec), rassoc(rassoc), unary(unary), func(func)
    {
    }
};

// clang-format off
static const Operator precTable[] =
{
    [ABrOpen]              = Operator(1,  0, 1, gen_subscript),
    [Plus]                 = Operator(4,  0, 0, gen_binop),
    [Minus]                = Operator(4,  0, 0, gen_binop),
    [Multiply]             = Operator(3,  0, 0, gen_binop),
    [Divide]               = Operator(3,  0, 0, gen_binop),
    [Modulo]               = Operator(3,  0, 0, gen_binop),
    [ShiftLeft]            = Operator(5,  0, 0, gen_binop),
    [ShiftRight]           = Operator(5,  0, 0, gen_binop),
    [LessThan]             = Operator(6,  0, 0, gen_compare),
    [GreaterThan]          = Operator(6,  0, 0, gen_compare),
    [LessThanEq]           = Operator(6,  0, 0, gen_compare),
    [GreaterThanEq]        = Operator(6,  0, 0, gen_compare),
    [Equals]               = Operator(7,  0, 0, gen_compare),
    [NotEquals]            = Operator(7,  0, 0, gen_compare),
    [BitwiseAND]           = Operator(8,  0, 0, gen_binop),
    [BitwiseOR]            = Operator(10, 0, 0, gen_binop),
    [BitwiseXOR]           = Operator(9,  0, 0, gen_binop),
    [LogicalAND]           = Operator(12, 0, 1, gen_logical),
    [LogicalOR]            = Operator(13, 0, 1, gen_logical),
    [Ternary]              = Operator(14, 1, 1, gen_ternary),
    [BitwiseConcat]        = Operator(11, 0, 0, gen_concat),
    [Assignment]           = Operator(15, 1, 0, gen_assign),
    [AssignmentAdd]        = Operator(15, 1, 0, gen_binop),
    [AssignmentSub]        = Operator(15, 1, 0, gen_binop),
    [AssignmentMul]        = Operator(15, 1, 0, gen_binop),
    [AssignmentDiv]        = Operator(15, 1, 0, gen_binop),
    [AssignmentMod]        = Operator(15, 1, 0, gen_binop),
    [AssignmentAND]        = Operator(15, 1, 0, gen_binop),
    [AssignmentOR]         = Operator(15, 1, 0, gen_binop),
    [AssignmentXOR]        = Operator(15, 1, 0, gen_binop),
    [AssignmentShiftRight] = Operator(15, 0, 0, gen_binop),
    [AssignmentShiftLeft]  = Operator(15, 0, 0, gen_binop),
    [Increment]            = Operator(1,  0, 1, gen_preinc),
    [Decrement]            = Operator(1,  0, 1, gen_preinc),
};
// clang-format on

Value ParseExpressionTerminal(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build)
{
    auto& ctx = func->getContext();
    switch (ts.Peek().type)
    {
        case Identifier:
        {
            auto t = ts.Pop();
            if (t.ident.str == "X")
            {
                pop_cur(ts, ABrOpen);
                auto ident = pop_cur(ts, Identifier).ident.str;
                pop_cur(ts, ABrClose);
                if (ident == "rd")
                    return {func->getArg(0), 32, false};
                if (ident == "rs1")
                    return {func->getArg(1), 32, false};
                if (ident == "rs2")
                    return {func->getArg(2), 32, false};
            }
            auto iter = immediateNames.find(t.ident.str);
            if (iter != immediateNames.end())
            {
                Value v = {func->getArg(3 + iter->second), false};
                // In LLVM logic, the immediate is actually 32 bits
                // (just small enough to fit into 6 signed/unsigned bits)
                v.bitWidth = 32;
                //curInstr->SetSignedImm(iter->second, false);
                return v;
            }
            // rd is true to skip "if (rd != 0)" checks.
            if (t.ident.str == "rd")
                return {llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1)};

            auto iter2 = variables.find(t.ident.idx);
            if (iter2 != variables.end())
                return iter2->getSecond().back().val;

            error(("undefined symbol: " + std::string(t.ident.str)).c_str(), ts);
        }
        case IntLiteral:
        {
            Token t = ts.Pop();
            return Value(llvm::ConstantInt::get(llvm::Type::getIntNTy(ctx, t.literal.bitLen), t.literal.value,
                                                t.literal.isSigned),
                         t.literal.isSigned);
        }
        case Minus:
        {
            ts.Pop();
            auto expr = ParseExpression(ts, func, build, 15 - 2);
            promote_lvalue(build, expr);
            expr.bitWidth++;
            fit_to_size(expr, build);
            expr.ll = build.CreateNeg(expr.ll);
            expr.isSigned = true;
            return expr;
        }
        case BitwiseNOT:
        {
            ts.Pop();
            auto expr = ParseExpression(ts, func, build, 15 - 2);
            promote_lvalue(build, expr);
            expr.ll = build.CreateNot(expr.ll);
            return expr;
        }
        case LogicalNOT:
        {
            ts.Pop();
            auto expr = ParseExpression(ts, func, build, 15 - 2);
            promote_lvalue(build, expr);
            expr.ll = build.CreateICmpEQ(expr.ll, llvm::ConstantInt::get(expr.ll->getType(), 0));
            return expr;
        }
        case Increment:
        case Decrement:
        {
            bool dec = ts.Peek().type == Decrement;
            ts.Pop();
            auto expr = ParseExpression(ts, func, build, 15 - 2);
            return gen_postinc(ts, func, build, dec ? Decrement : Increment, expr, Value());
        }
        case RBrOpen:
        {
            ts.Pop();
            // Cast
            if (ts.Peek().type == SignedKeyword || ts.Peek().type == UnsignedKeyword)
            {
                bool isSigned = (ts.Pop().type == SignedKeyword);

                int len = 0;
                if (pop_cur_if(ts, LessThan))
                {
                    len = pop_cur(ts, IntLiteral).literal.value;
                    if (len == 0)
                        syntax_error(ts);
                    pop_cur(ts, GreaterThan);
                }
                pop_cur(ts, RBrClose);

                auto v = ParseExpression(ts, func, build, 15 - 2);
                
                // Keep track locally if the immediate is used 
                // signed or unsigned for this instruction.
                // Using both is undefined behaviour.
                for (size_t i = 0; i < IMM_CNT; i++)
                    if (v.ll == func->getArg(i + 3) && (len == 0 || len == 32))
                        if (!curInstr->SetSignedImm(i, isSigned)) error("internal errro", ts);

                if (len != 0)
                {
                    promote_lvalue(build, v);
                    auto newType = llvm::Type::getIntNTy(ctx, len);

                    if (v.bitWidth > len)
                        v.ll = build.CreateTrunc(v.ll, newType);
                    if (v.bitWidth < len)
                    {
                        if (v.isSigned)
                            v.ll = build.CreateSExt(v.ll, newType);
                        else
                            v.ll = build.CreateZExt(v.ll, newType);
                    }
                    v.bitWidth = len;
                }
                v.isSigned = isSigned;
                return v;
            }
            else
            {
                auto rv = ParseExpression(ts, func, build);
                pop_cur(ts, RBrClose);
                return rv;
            }
            break;
        }
        default: syntax_error(ts);
    }
}

Value ParseExpression(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build, int minPrec)
{
    Value left = ParseExpressionTerminal(ts, func, build);

    while (true)
    {
        TokenType tt = ts.Peek().type;
        if (!(tt >= OP_START && tt <= OP_END))
            return left;

        Operator const op = precTable[tt - OP_START];
        if (op.prec < minPrec)
            return left;
        if (op.func == nullptr)
            not_implemented(ts);

        ts.Pop();

        Value right;
        if (!op.unary)
            right = ParseExpression(ts, func, build, op.prec + (op.rassoc ? 0 : 1));
        left = op.func(ts, func, build, tt, left, right);
    }
}

void ParseDeclaration (TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build)
{
    auto& ctx = func->getContext();

    bool sgn = pop_cur_if(ts, SignedKeyword);
    if (!sgn) pop_cur(ts, UnsignedKeyword);
    
    int bitSize = -1;
    if (pop_cur_if(ts, LessThan))
    {
        bitSize = pop_cur(ts, IntLiteral).literal.value;
        if (bitSize <= 0) error("invalid size", ts);
        pop_cur(ts, GreaterThan);
    }
    auto ident = pop_cur(ts, Identifier).ident;
    uint32_t id = ident.idx;
    
    std::optional<Value> init;
    if (pop_cur_if(ts, Assignment))
    {
        init = ParseExpression(ts, func, build);
        promote_lvalue(build, init.value());
        if (bitSize == -1) bitSize = init->bitWidth;
        if (init->bitWidth > bitSize) error("implicit truncation", ts);
        init->bitWidth = bitSize;
        fit_to_size(init.value(), build);
    }

    Value v;
    v.bitWidth = bitSize;
    v.isSigned = sgn;
    v.isLValue = true;
    {
        llvm::BasicBlock* cur = build.GetInsertBlock();
        if (entry->getTerminator() == nullptr)
            build.SetInsertPoint(entry);
        else
            build.SetInsertPoint(entry->getTerminator());
        v.ll = build.CreateAlloca(llvm::Type::getIntNTy(ctx, ceil_to_pow2(bitSize)));
        build.SetInsertPoint(cur);
    }

    if (init.has_value())
        build.CreateStore(init->ll, v.ll, false);
    
    if (!variables[id].empty() && variables[id].back().scope == scopeDepth)
        error(("redefinition: " + std::string(ident.str)).c_str(), ts);

    variables[id].push_back((Variable){v, scopeDepth});

    pop_cur(ts, Semicolon);
}

void ParseStatement(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build)
{
    auto& ctx = func->getContext();

    switch (ts.Peek().type)
    {
        case IfKeyword:
        {
            ts.Pop();
            pop_cur(ts, RBrOpen);
            Value cond = ParseExpression(ts, func, build);
            pop_cur(ts, RBrClose);

            llvm::Value* condBool = build.CreateICmpNE(cond.ll, llvm::ConstantInt::get(cond.ll->getType(), 0));
            llvm::BasicBlock* bbTrue = llvm::BasicBlock::Create(ctx, "", func);
            llvm::BasicBlock* bbFalse = llvm::BasicBlock::Create(ctx, "", func);
            build.CreateCondBr(condBool, bbTrue, bbFalse);

            build.SetInsertPoint(bbTrue);
            ParseStatement(ts, func, build);

            if (pop_cur_if(ts, ElseKeyword))
            {
                llvm::BasicBlock* bbCont = llvm::BasicBlock::Create(ctx, "", func);
                build.CreateBr(bbCont);
                build.SetInsertPoint(bbFalse);
                ParseStatement(ts, func, build);
                build.CreateBr(bbCont);
                build.SetInsertPoint(bbCont);
            }
            else
            {
                build.CreateBr(bbFalse);
                build.SetInsertPoint(bbFalse);
            }
            break;
        }
        case WhileKeyword:
        case ForKeyword:
        {
            llvm::BasicBlock* bbHdr = llvm::BasicBlock::Create(ctx, "loop_hdr", func);
            llvm::BasicBlock* bbBody = llvm::BasicBlock::Create(ctx, "loop_body", func);
            llvm::BasicBlock* bbBreak = llvm::BasicBlock::Create(ctx, "loop_break", func);
            llvm::BasicBlock* bbInc = nullptr;
            
            bool isFor = ts.Pop().type == ForKeyword;
            pop_cur(ts, RBrOpen);
            if (isFor)
            {
                create_var_scope();
                if (!pop_cur_if(ts, Semicolon))
                    ParseDeclaration(ts, func, build);
            }

            build.CreateBr(bbHdr);
            build.SetInsertPoint(bbHdr);

            Value cond = ParseExpression(ts, func, build);
            if (isFor)
            {
                bbInc = llvm::BasicBlock::Create(ctx, "loop_inc", func);
                pop_cur(ts, Semicolon);
                build.SetInsertPoint(bbInc);
                if (!pop_cur_if(ts, Semicolon))
                    ParseExpression(ts, func, build);
                build.CreateBr(bbHdr);
                build.SetInsertPoint(bbHdr);
            }
            pop_cur(ts, RBrClose);

            llvm::Value* condBool = build.CreateICmpEQ(cond.ll, llvm::ConstantInt::get(cond.ll->getType(), 0));
            build.CreateCondBr(condBool, bbBreak, bbBody);
            build.SetInsertPoint(bbBody);

            ParseStatement(ts, func, build);

            build.CreateBr(isFor ? bbInc : bbHdr);
            build.SetInsertPoint(bbBreak);

            if (isFor) pop_var_scope();
            break;
        }
        case UnsignedKeyword:
        case SignedKeyword:
        {
            ParseDeclaration(ts, func, build);
            break;
        }
        case CBrOpen: ParseScope(ts, func, build); break;
        default:
        {
            ParseExpression(ts, func, build);
            pop_cur(ts, Semicolon);
            break;
        }
    }
}

void ParseScope(TokenStream& ts, llvm::Function* func, llvm::IRBuilder<>& build)
{
    pop_cur(ts, CBrOpen);
    create_var_scope();

    while (ts.Peek().type != CBrClose)
        ParseStatement(ts, func, build);
    
    pop_var_scope();
    pop_cur(ts, CBrClose);
}



void ParseEncoding (TokenStream& ts, CDSLInstr& instr)
{   
    pop_cur(ts, EncodingKeyword);
    pop_cur(ts, Colon);

    uint offset = 32;
    
    std::map<std::string_view, int> idents = {{"rd", 0}, {"rs1", 1}, {"rs2", 2}};
    while (1)
    {

        switch (ts.Peek().type)
        {
            case IntLiteral:
            {
                auto litT = ts.Pop();
                uint len = litT.literal.bitLen;
                offset -= len;
                instr.fields.push_back(CDSLInstr::Field::ConstField(litT.literal.value, offset, len));
                break;
            }
            case Identifier:
            {
                auto idT = ts.Pop();
                
                pop_cur(ts, ABrOpen);
                auto hi = pop_cur(ts, IntLiteral).literal.value;
                auto lo = hi;
                if (pop_cur_if(ts, Colon))
                    lo = pop_cur(ts, IntLiteral).literal.value;
                pop_cur(ts, ABrClose);

                if (lo > hi || lo > 31 || hi > 31) error("out of bounds", ts);

                auto len =  (hi - lo + 1);
                offset -= len;

                auto match = idents.find(idT.ident.str);
                if (match == idents.end())
                {
                    if (idents.size() >= 5) error("too many immediate definitions", ts);
                    match = idents.insert(std::make_pair(idT.ident.str, (int)idents.size())).first;
                    immediateNames.insert(std::make_pair(llvm::StringRef(idT.ident.str), immediateNames.size()));
                }
                uint32_t immSize = 0;
                if (match->second > 2)
                {
                    auto sizeBegin = std::find_if(idT.ident.str.begin(), idT.ident.str.end(), isdigit);
                    if (sizeBegin == idT.ident.str.end()) error("immediate does not specify size", ts);
                    immSize = std::stol(sizeBegin, nullptr, 10);
                    if (immSize > 31) error("invalid immediate size", ts);
                }
                
                instr.fields.push_back(CDSLInstr::Field::RegImmField(match->second, offset, lo, len, false, immSize));
                break;
            }
            default: syntax_error(ts);
        }
        if (pop_cur_if(ts, Semicolon))
        {
            if (offset != 0) error("instruction length is not 32 bits", ts);
            return;
        }
        pop_cur(ts, BitwiseConcat);
    }
}

void ParseArguments (TokenStream& ts, CDSLInstr& instr)
{
    pop_cur(ts, AssemblyKeyword);
    pop_cur(ts, Colon);
    
    // "$rd, $rs1, $rs2, $imm5"
    auto str = std::string(pop_cur(ts, StringLiteral).strLit.str);
    str = std::regex_replace(str, std::regex("\\{name\\(rd\\)\\}"), "$rd");
    str = std::regex_replace(str, std::regex("\\{name\\(rs1\\)\\}"), "$rs1");
    str = std::regex_replace(str, std::regex("\\{name\\(rs2\\)\\}"), "$rs2");
    
    for (auto const& imm : immediateNames)
        str = std::regex_replace(str, std::regex(("\\{" + imm.first + "\\}").str()), "$imm" + (imm.second ? std::to_string(imm.second + 1) : ""));

    instr.argString = str;
    pop_cur(ts, Semicolon);
}

void ParseBehaviour (TokenStream& ts, CDSLInstr& instr, llvm::Module* mod, Token const& ident)
{
    auto& ctx = mod->getContext();
    auto ptrT = llvm::PointerType::get(ctx, 0);
    auto immT = llvm::Type::getInt32Ty(ctx);
    auto fType = llvm::FunctionType::get(llvm::Type::getVoidTy(ctx), {ptrT, ptrT, ptrT, immT, immT}, false);

    pop_cur(ts, BehaviorKeyword);
    pop_cur(ts, Colon);

    llvm::Function* func =
        llvm::Function::Create(fType, llvm::GlobalValue::ExternalLinkage,
                                std::string("impl") + std::string(ident.ident.str), mod);
    
    const std::array<std::string, 5> argNames = {"rd", "rs1", "rs2", "imm", "imm2"};
    for (size_t i = 0; i < argNames.size(); i++)
        func->getArg(i)->setName(argNames[i]);
    
    // For vectorization to work, we must assume that
    // the destination does not overlap with sources.
    // For simulators using this generated code, this means
    // that rd has to be a pointer to a temporary variable.
    func->getArg(0)->addAttr(llvm::Attribute::NoAlias);
    entry = llvm::BasicBlock::Create(ctx, "", func);
    llvm::IRBuilder<> build(entry);

    ParseScope(ts, func, build);
    build.CreateRetVoid();
}

std::vector<CDSLInstr> ParseCoreDSL2(TokenStream& ts, llvm::Module* mod)
{
    std::vector<CDSLInstr> instrs;

    // boilerplate
    pop_cur(ts, Identifier);
    pop_cur(ts, Identifier);
    if (pop_cur_if(ts, ExtendsKeyword))
        pop_cur(ts, Identifier);
    pop_cur(ts, CBrOpen);
    pop_cur(ts, InstructionsKeyword);
    pop_cur(ts, CBrOpen);

    while (ts.Peek().type != CBrClose)
    {
        reset_globals();
        Token ident = pop_cur(ts, Identifier);
        pop_cur(ts, CBrOpen);
        CDSLInstr instr{.name = std::string(ident.ident.str)};
        curInstr = &instr;

        while (ts.Peek().type != CBrClose)
        {
            switch (ts.Peek().type)
            {
                case EncodingKeyword:
                    ParseEncoding(ts, instr);
                    break;
                case AssemblyKeyword:
                    ParseArguments(ts, instr);
                    break;
                case BehaviorKeyword:
                    ParseBehaviour(ts, instr, mod, ident);
                    break;
                default: syntax_error(ts);
            }
        }
        pop_cur(ts, CBrClose);
        instrs.push_back(instr); 
    }

    pop_cur(ts, CBrClose);
    pop_cur(ts, CBrClose);

    return instrs;
}
