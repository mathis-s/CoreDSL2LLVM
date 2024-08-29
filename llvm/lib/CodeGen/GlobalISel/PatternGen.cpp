//===- llvm/CodeGen/GlobalISel/PatternGen.cpp - PatternGen ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the PatternGen class.
//===----------------------------------------------------------------------===//

/*
Helpful list from: lvm-project/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td

Warning: dropped special conditions and preds!

handled generic instructions are marked with a >
TODOs are marked with a !
Maybes are marked with a ?
FPs are marked with a %
Ignores are marked with a #

Missing in map: G_SHUFFLE_VECTOR (?)

>def : GINodeEquiv<G_ANYEXT, anyext>;
>def : GINodeEquiv<G_SEXT, sext>;
>def : GINodeEquiv<G_ZEXT, zext>;
>def : GINodeEquiv<G_TRUNC, trunc>;
>def : GINodeEquiv<G_BITCAST, bitconvert>;
// G_INTTOPTR - SelectionDAG has no equivalent.
// G_PTRTOINT - SelectionDAG has no equivalent.
> def : GINodeEquiv<G_CONSTANT, imm>;
// timm must not be materialized and therefore has no GlobalISel equivalent
%def : GINodeEquiv<G_FCONSTANT, fpimm>;
>def : GINodeEquiv<G_IMPLICIT_DEF, undef>;
#def : GINodeEquiv<G_FRAME_INDEX, frameindex>;
#def : GINodeEquiv<G_BLOCK_ADDR, blockaddress>;
>def : GINodeEquiv<G_PTR_ADD, ptradd>;
>def : GINodeEquiv<G_ADD, add>;
>def : GINodeEquiv<G_SUB, sub>;
>def : GINodeEquiv<G_MUL, mul>;
>def : GINodeEquiv<G_UMULH, mulhu>;
>def : GINodeEquiv<G_SMULH, mulhs>;
>def : GINodeEquiv<G_SDIV, sdiv>;
>def : GINodeEquiv<G_UDIV, udiv>;
>def : GINodeEquiv<G_SREM, srem>;
>def : GINodeEquiv<G_UREM, urem>;
>def : GINodeEquiv<G_AND, and>;
>def : GINodeEquiv<G_OR, or>;
>def : GINodeEquiv<G_XOR, xor>;
>def : GINodeEquiv<G_SHL, shl>;
>def : GINodeEquiv<G_LSHR, srl>;
>def : GINodeEquiv<G_ASHR, sra>;
>def : GINodeEquiv<G_SADDSAT, saddsat>;
>def : GINodeEquiv<G_UADDSAT, uaddsat>;
>def : GINodeEquiv<G_SSUBSAT, ssubsat>;
>def : GINodeEquiv<G_USUBSAT, usubsat>;
>def : GINodeEquiv<G_SSHLSAT, sshlsat>;
>def : GINodeEquiv<G_USHLSAT, ushlsat>;
>def : GINodeEquiv<G_SMULFIX, smulfix>;
>def : GINodeEquiv<G_UMULFIX, umulfix>;
>def : GINodeEquiv<G_SMULFIXSAT, smulfixsat>;
>def : GINodeEquiv<G_UMULFIXSAT, umulfixsat>;
>def : GINodeEquiv<G_SDIVFIX, sdivfix>;
>def : GINodeEquiv<G_UDIVFIX, udivfix>;
>def : GINodeEquiv<G_SDIVFIXSAT, sdivfixsat>;
>def : GINodeEquiv<G_UDIVFIXSAT, udivfixsat>;
>def : GINodeEquiv<G_SELECT, select>;
?def : GINodeEquiv<G_SELECT, vselect>;
%def : GINodeEquiv<G_FNEG, fneg>;
%def : GINodeEquiv<G_FPEXT, fpextend>;
%def : GINodeEquiv<G_FPTRUNC, fpround>;
%def : GINodeEquiv<G_FPTOSI, fp_to_sint>;
%def : GINodeEquiv<G_FPTOUI, fp_to_uint>;
%def : GINodeEquiv<G_SITOFP, sint_to_fp>;
%def : GINodeEquiv<G_UITOFP, uint_to_fp>;
%def : GINodeEquiv<G_FADD, fadd>;
%def : GINodeEquiv<G_FSUB, fsub>;
%def : GINodeEquiv<G_FMA, fma>;
%def : GINodeEquiv<G_FMAD, fmad>;
%def : GINodeEquiv<G_FMUL, fmul>;
%def : GINodeEquiv<G_FDIV, fdiv>;
%def : GINodeEquiv<G_FREM, frem>;
%def : GINodeEquiv<G_FPOW, fpow>;
%def : GINodeEquiv<G_FEXP2, fexp2>;
%def : GINodeEquiv<G_FEXP10, fexp10>;
%def : GINodeEquiv<G_FLOG2, flog2>;
%def : GINodeEquiv<G_FLDEXP, fldexp>;
%def : GINodeEquiv<G_FCANONICALIZE, fcanonicalize>;
%def : GINodeEquiv<G_IS_FPCLASS, is_fpclass>;
#def : GINodeEquiv<G_INTRINSIC, intrinsic_wo_chain>;
%def : GINodeEquiv<G_GET_FPENV, get_fpenv>;
%def : GINodeEquiv<G_SET_FPENV, set_fpenv>;
%def : GINodeEquiv<G_RESET_FPENV, reset_fpenv>;
%def : GINodeEquiv<G_GET_FPMODE, get_fpmode>;
%def : GINodeEquiv<G_SET_FPMODE, set_fpmode>;
%def : GINodeEquiv<G_RESET_FPMODE, reset_fpmode>;
#def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_void>;
#def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_w_chain>;
?def : GINodeEquiv<G_BR, br>;
?def : GINodeEquiv<G_BRCOND, brcond>;
>def : GINodeEquiv<G_BSWAP, bswap>;
>def : GINodeEquiv<G_BITREVERSE, bitreverse>;
>def : GINodeEquiv<G_FSHL, fshl>;
>def : GINodeEquiv<G_FSHR, fshr>;
>def : GINodeEquiv<G_CTLZ, ctlz>;
>def : GINodeEquiv<G_CTTZ, cttz>;
>def : GINodeEquiv<G_CTLZ_ZERO_UNDEF, ctlz_zero_undef>;
>def : GINodeEquiv<G_CTTZ_ZERO_UNDEF, cttz_zero_undef>;
>def : GINodeEquiv<G_CTPOP, ctpop>;
>def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, extractelt>;
>def : GINodeEquiv<G_INSERT_VECTOR_ELT, vector_insert>;
?def : GINodeEquiv<G_CONCAT_VECTORS, concat_vectors>;
>def : GINodeEquiv<G_BUILD_VECTOR, build_vector>;
%def : GINodeEquiv<G_FCEIL, fceil>;
%def : GINodeEquiv<G_FCOS, fcos>;
%def : GINodeEquiv<G_FSIN, fsin>;
%def : GINodeEquiv<G_FTAN, ftan>;
%def : GINodeEquiv<G_FACOS, facos>;
%def : GINodeEquiv<G_FASIN, fasin>;
%def : GINodeEquiv<G_FATAN, fatan>;
%def : GINodeEquiv<G_FCOSH, fcosh>;
%def : GINodeEquiv<G_FSINH, fsinh>;
%def : GINodeEquiv<G_FTANH, ftanh>;
%def : GINodeEquiv<G_FABS, fabs>;
%def : GINodeEquiv<G_FSQRT, fsqrt>;
%def : GINodeEquiv<G_FFLOOR, ffloor>;
%def : GINodeEquiv<G_FRINT, frint>;
%def : GINodeEquiv<G_FNEARBYINT, fnearbyint>;
%def : GINodeEquiv<G_INTRINSIC_TRUNC, ftrunc>;
%def : GINodeEquiv<G_INTRINSIC_ROUND, fround>;
%def : GINodeEquiv<G_INTRINSIC_ROUNDEVEN, froundeven>;
#def : GINodeEquiv<G_INTRINSIC_LRINT, lrint>;
#def : GINodeEquiv<G_INTRINSIC_LLRINT, llrint>;
%def : GINodeEquiv<G_FCOPYSIGN, fcopysign>;
>def : GINodeEquiv<G_SMIN, smin>;
>def : GINodeEquiv<G_SMAX, smax>;
>def : GINodeEquiv<G_UMIN, umin>;
>def : GINodeEquiv<G_UMAX, umax>;
>def : GINodeEquiv<G_ABS, abs>;
%def : GINodeEquiv<G_FMINNUM, fminnum>;
%def : GINodeEquiv<G_FMAXNUM, fmaxnum>;
%def : GINodeEquiv<G_FMINNUM_IEEE, fminnum_ieee>;
%def : GINodeEquiv<G_FMAXNUM_IEEE, fmaxnum_ieee>;
%def : GINodeEquiv<G_FMAXIMUM, fmaximum>;
%def : GINodeEquiv<G_FMINIMUM, fminimum>;
#def : GINodeEquiv<G_READCYCLECOUNTER, readcyclecounter>;
#def : GINodeEquiv<G_READSTEADYCOUNTER, readsteadycounter>;
>def : GINodeEquiv<G_ROTR, rotr>;
>def : GINodeEquiv<G_ROTL, rotl>;
?def : GINodeEquiv<G_LROUND, lround>;
?def : GINodeEquiv<G_LLROUND, llround>;
%def : GINodeEquiv<G_VECREDUCE_FADD, vecreduce_fadd>;
%def : GINodeEquiv<G_VECREDUCE_FMAX, vecreduce_fmax>;
%def : GINodeEquiv<G_VECREDUCE_FMIN, vecreduce_fmin>;
%def : GINodeEquiv<G_VECREDUCE_FMAXIMUM, vecreduce_fmaximum>;
%def : GINodeEquiv<G_VECREDUCE_FMINIMUM, vecreduce_fminimum>;
?def : GINodeEquiv<G_VECREDUCE_UMIN, vecreduce_umin>;
?def : GINodeEquiv<G_VECREDUCE_UMAX, vecreduce_umax>;
?def : GINodeEquiv<G_VECREDUCE_SMIN, vecreduce_smin>;
?def : GINodeEquiv<G_VECREDUCE_SMAX, vecreduce_smax>;
>def : GINodeEquiv<G_VECREDUCE_ADD, vecreduce_add>;
?def : GINodeEquiv<G_VECTOR_COMPRESS, vector_compress>;
%def : GINodeEquiv<G_STRICT_FADD, strict_fadd>;
%def : GINodeEquiv<G_STRICT_FSUB, strict_fsub>;
%def : GINodeEquiv<G_STRICT_FMUL, strict_fmul>;
%def : GINodeEquiv<G_STRICT_FDIV, strict_fdiv>;
%def : GINodeEquiv<G_STRICT_FREM, strict_frem>;
%def : GINodeEquiv<G_STRICT_FMA, strict_fma>;
%def : GINodeEquiv<G_STRICT_FSQRT, strict_fsqrt>;
%def : GINodeEquiv<G_STRICT_FLDEXP, strict_fldexp>;
>def : GINodeEquiv<G_LOAD, ld>;
>def : GINodeEquiv<G_ICMP, setcc>;
>def : GINodeEquiv<G_STORE, st>;
#def : GINodeEquiv<G_STORE, atomic_store>;
#def : GINodeEquiv<G_LOAD, atomic_load>;
#def : GINodeEquiv<G_ATOMIC_CMPXCHG, atomic_cmp_swap>;
#def : GINodeEquiv<G_ATOMICRMW_XCHG, atomic_swap>;
#def : GINodeEquiv<G_ATOMICRMW_ADD, atomic_load_add>;
#def : GINodeEquiv<G_ATOMICRMW_SUB, atomic_load_sub>;
#def : GINodeEquiv<G_ATOMICRMW_AND, atomic_load_and>;
#def : GINodeEquiv<G_ATOMICRMW_NAND, atomic_load_nand>;
#def : GINodeEquiv<G_ATOMICRMW_OR, atomic_load_or>;
#def : GINodeEquiv<G_ATOMICRMW_XOR, atomic_load_xor>;
#def : GINodeEquiv<G_ATOMICRMW_MIN, atomic_load_min>;
#def : GINodeEquiv<G_ATOMICRMW_MAX, atomic_load_max>;
#def : GINodeEquiv<G_ATOMICRMW_UMIN, atomic_load_umin>;
#def : GINodeEquiv<G_ATOMICRMW_UMAX, atomic_load_umax>;
#def : GINodeEquiv<G_ATOMICRMW_FADD, atomic_load_fadd>;
#def : GINodeEquiv<G_ATOMICRMW_FSUB, atomic_load_fsub>;
#def : GINodeEquiv<G_ATOMICRMW_FMAX, atomic_load_fmax>;
#def : GINodeEquiv<G_ATOMICRMW_FMIN, atomic_load_fmin>;
#def : GINodeEquiv<G_ATOMICRMW_UINC_WRAP, atomic_load_uinc_wrap>;
#def : GINodeEquiv<G_ATOMICRMW_UDEC_WRAP, atomic_load_udec_wrap>;
#def : GINodeEquiv<G_FENCE, atomic_fence>;
#def : GINodeEquiv<G_PREFETCH, prefetch>;
#def : GINodeEquiv<G_TRAP, trap>;
#def : GINodeEquiv<G_DEBUGTRAP, debugtrap>;
#def : GINodeEquiv<G_UBSANTRAP, ubsantrap>;
*/

#include "llvm/CodeGen/GlobalISel/PatternGen.h"
#include "../../../tools/pattern-gen/lib/InstrInfo.hpp"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/CodeGenTypes/LowLevelType.h"
#include "llvm/Config/config.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CodeGenCoverage.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/PredicateInfo.h"
#include "llvm/ADT/Statistic.h"
#include <memory>
#include <sstream>
#include <string>
#include <utility>

#define DEBUG_TYPE "pattern-gen"

using namespace llvm;

STATISTIC(
    PatternGenNumInstructionsProcessed,
    "Processed instructions");
STATISTIC(
    PatternGenNumInstructionsFailing,
    "Failing instructions");
STATISTIC(
    PatternGenNumPatternsGenerated,
    "Generated patterns");
STATISTIC(
    PatternGenNumPatternsFailing,
    "Failing patterns");
STATISTIC(
    PatternGenNumErrorMultipleBlocks,
    "Errors of type: MULTIPLE_BLOCKS");
STATISTIC(
    PatternGenNumErrorFormatReturn,
    "Errors of type: FORMAT_RETURN");
STATISTIC(
    PatternGenNumErrorFormatStore,
    "Errors of type: FORMAT_STORE");
STATISTIC(
    PatternGenNumErrorFormatLoad,
    "Errors of type: FORMAT_LOAD");
STATISTIC(
    PatternGenNumErrorFormatImm,
    "Errors of type: FORMAT_IMM");
STATISTIC(
    PatternGenNumErrorFormat,
    "Errors of type: FORMAT");

#ifdef LLVM_GISEL_COV_PREFIX
static cl::opt<std::string>
    CoveragePrefix("gisel-coverage-prefix", cl::init(LLVM_GISEL_COV_PREFIX),
                   cl::desc("Record GlobalISel rule coverage files of this "
                            "prefix if instrumentation was generated"));
#else
static const std::string CoveragePrefix;
#endif

std::ostream *PatternGenArgs::OutStream = nullptr;
std::vector<CDSLInstr> const *PatternGenArgs::Instrs = nullptr;
PGArgsStruct PatternGenArgs::Args;

struct PatternArg {
  std::string ArgTypeStr;
  LLT Llt;
  // We also have in and out bits in the CDSLInstr struct itself.
  // These bits are currently ignored though. Instead, we find inputs
  // and outputs during pattern gen and store that in these fields.
  // We may want to add a warning on mismatch between the two.
  bool In;
  bool Out;
};

static CDSLInstr const *CurInstr = nullptr;
static SmallVector<PatternArg, 8> PatternArgs;
static bool MayLoad = 0;
static bool MayStore = 0;

static uint64_t XLen;
static std::string RegT;

char PatternGen::ID = 0;
INITIALIZE_PASS_BEGIN(
    PatternGen, DEBUG_TYPE,
    "Convert instruction behavior functions to TableGen ISel patterns", false,
    false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_DEPENDENCY(GISelKnownBitsAnalysis)
INITIALIZE_PASS_DEPENDENCY(ProfileSummaryInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(LazyBlockFrequencyInfoPass)
INITIALIZE_PASS_END(
    PatternGen, DEBUG_TYPE,
    "Convert instruction behavior functions to TableGen ISel patterns", false,
    false)

PatternGen::PatternGen(CodeGenOptLevel OL)
    : MachineFunctionPass(ID), OptLevel(OL) {}

// In order not to crash when calling getAnalysis during testing with -run-pass
// we use the default opt level here instead of None, so that the addRequired()
// calls are made in getAnalysisUsage().
PatternGen::PatternGen()
    : MachineFunctionPass(ID), OptLevel(CodeGenOptLevel::Default) {}

void PatternGen::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<TargetPassConfig>();
  AU.addRequired<GISelKnownBitsAnalysis>();
  AU.addPreserved<GISelKnownBitsAnalysis>();

  if (OptLevel != CodeGenOptLevel::None) {
    AU.addRequired<ProfileSummaryInfoWrapperPass>();
    LazyBlockFrequencyInfoPass::getLazyBFIAnalysisUsage(AU);
  }
  getSelectionDAGFallbackAnalysisUsage(AU);
  MachineFunctionPass::getAnalysisUsage(AU);
}

enum PatternErrorT {
  SUCCESS = 0,
  MULTIPLE_BLOCKS,
  FORMAT_RETURN,
  FORMAT_STORE,
  FORMAT_LOAD,
  FORMAT_IMM,
  FORMAT
};
struct PatternError {
  PatternErrorT Type;
  MachineInstr *Inst;
  PatternError(PatternErrorT Type) : Type(Type), Inst(nullptr) {}
  PatternError(PatternErrorT Type, MachineInstr *Inst)
      : Type(Type), Inst(Inst) {}
  operator bool() const { return Type != 0; }
};

std::string Errors[] = {"success",        "multiple blocks", "expected return",
                        "expected store", "load format",     "immediate format",
                        "format"};
llvm::Statistic* ErrorStats[] = {
  &PatternGenNumErrorMultipleBlocks,
  &PatternGenNumErrorFormatReturn,
  &PatternGenNumErrorFormatStore,
  &PatternGenNumErrorFormatLoad,
  &PatternGenNumErrorFormatImm,
  &PatternGenNumErrorFormat,
};

static const std::unordered_map<unsigned, std::string> CmpStr = {
    {CmpInst::Predicate::ICMP_EQ, "SETEQ"},
    {CmpInst::Predicate::ICMP_NE, "SETNE"},
    {CmpInst::Predicate::ICMP_SLT, "SETLT"},
    {CmpInst::Predicate::ICMP_SLE, "SETLE"},
    {CmpInst::Predicate::ICMP_SGT, "SETGT"},
    {CmpInst::Predicate::ICMP_SGE, "SETGE"},
    {CmpInst::Predicate::ICMP_ULT, "SETULT"},
    {CmpInst::Predicate::ICMP_ULE, "SETULE"},
    {CmpInst::Predicate::ICMP_UGT, "SETUGT"},
    {CmpInst::Predicate::ICMP_UGE, "SETUGE"},
};

std::string lltToString(LLT Llt) {
  if (Llt.isFixedVector())
    return "v" + std::to_string(Llt.getElementCount().getFixedValue()) +
           lltToString(Llt.getElementType());
  if (Llt.isScalar())
    return "i" + std::to_string(Llt.getSizeInBits());
  assert(0 && "invalid type");
  return "invalid";
}

std::string lltToRegTypeStr(LLT Type) {
  if (Type.isValid()) {
    if (Type.isFixedVector() && Type.getElementType().isScalar() &&
        Type.getSizeInBits() == 32) {
      if (Type.getElementType().getSizeInBits() == 8)
        return "GPR32V4";
      if (Type.getElementType().getSizeInBits() == 16)
        return "GPR32V2";
      abort();
    } else
      return "GPR";
  }
  assert(0 && "invalid type");
  return "invalid";
}

std::string makeImmTypeStr(int Size, bool Signed) {
  return (Signed ? "simm" : "uimm") + std::to_string(Size);
}

struct PatternNode {
  enum PatternNodeKind {
    PN_NOp,
    PN_Binop,
    PN_Ternop,
    PN_Shuffle,
    PN_Compare,
    PN_Unop,
    PN_Constant,
    PN_Register,
    PN_Select,
  };

private:
  const PatternNodeKind Kind;

public:
  PatternNodeKind getKind() const { return Kind; }
  LLT Type;
  bool IsImm = false;
  PatternNode(PatternNodeKind Kind, LLT Type, bool IsImm) : Kind(Kind), Type(Type), IsImm(IsImm) {}

  virtual std::string patternString(int Indent = 0) = 0;
  virtual LLT getRegisterTy(int OperandId) const {
    if (OperandId == -1)
      return Type;
    return LLT();
  }
  virtual ~PatternNode() {}
};

struct NOpNode : public PatternNode {
  int Op;
  std::vector<std::unique_ptr<PatternNode>> Operands;
  NOpNode(LLT Type, int Op, std::vector<std::unique_ptr<PatternNode>> Operands)
      : PatternNode(PN_NOp, Type, false), Op(Op), Operands(std::move(Operands)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> NOpStr = {
        {TargetOpcode::G_BUILD_VECTOR, "build_vector"},
        {TargetOpcode::G_SELECT, "vselect"}};

    std::string S = "(" + std::string(NOpStr.at(Op)) + " ";
    for (auto &Operand : Operands)
      S += Operand->patternString(Indent + 1) + ", ";
    if (!Operands.empty())
      S = S.substr(0, S.size() - 2);

    S += ")";
    return S;
  }
  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;

    for (auto &Operand : Operands) {
      auto T = Operand->getRegisterTy(OperandId);
      if (T.isValid())
        return T;
    }
    return LLT();
  }
  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_NOp;
  }
};

struct ShuffleNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> First;
  std::unique_ptr<PatternNode> Second;
  ArrayRef<int> Mask;

  ShuffleNode(LLT Type, int Op, std::unique_ptr<PatternNode> First,
              std::unique_ptr<PatternNode> Second, ArrayRef<int> Mask)
      : PatternNode(PN_Shuffle, Type, false), Op(Op), First(std::move(First)),
        Second(std::move(Second)), Mask(std::move(Mask)) {}

  std::string patternString(int Indent = 0) override {
    std::string TypeStr = lltToString(Type);
    std::string MaskStr = "";

    for (size_t I = 0; I < Mask.size(); I++) {
      if (I != 0) {
        MaskStr += ", ";
      }
      MaskStr += std::to_string(Mask[I]);
    }
    std::string OpString = "(vector_shuffle<" + MaskStr + "> " +
                           First->patternString(Indent + 1) + ", " +
                           Second->patternString(Indent + 1) + ")";

    // Explicitly specifying types for all ops increases pattern compile time
    // significantly, so we only do for ops where deduction fails otherwise.
    bool PrintType = false;

    if (PrintType)
      return "(" + TypeStr + " " + OpString + ")";
    return OpString;
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;

    auto FirstT = First->getRegisterTy(OperandId);
    auto SecondT = Second->getRegisterTy(OperandId);
    return FirstT.isValid() ? FirstT : SecondT;
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Shuffle;
  }
};

struct TernopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> First;
  std::unique_ptr<PatternNode> Second;
  std::unique_ptr<PatternNode> Third;

  TernopNode(LLT Type, int Op, std::unique_ptr<PatternNode> First,
             std::unique_ptr<PatternNode> Second,
             std::unique_ptr<PatternNode> Third)
      : PatternNode(PN_Ternop, Type, false), Op(Op), First(std::move(First)),
        Second(std::move(Second)), Third(std::move(Third)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> TernopStr = {
        {TargetOpcode::G_FSHL, "fshl"},
        {TargetOpcode::G_FSHR, "fshr"},
        {TargetOpcode::G_INSERT_VECTOR_ELT, "vector_insert"},
        {TargetOpcode::G_SELECT, "select"}};

    std::string TypeStr = lltToString(Type);
    std::string OpString = "(" + std::string(TernopStr.at(Op)) + " " +
                           First->patternString(Indent + 1) + ", " +
                           Second->patternString(Indent + 1) + ", " +
                           Third->patternString(Indent + 1) + ")";

    bool PrintType = false;
    if (PrintType)
      return "(" + TypeStr + " " + OpString + ")";
    return OpString;
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;

    auto FirstT = First->getRegisterTy(OperandId);
    auto SecondT = Second->getRegisterTy(OperandId);
    auto ThirdT = Third->getRegisterTy(OperandId);
    return FirstT.isValid() ? FirstT : (SecondT.isValid() ? SecondT : ThirdT);
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Ternop;
  }
};

struct BinopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Left;
  std::unique_ptr<PatternNode> Right;

  BinopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Left,
            std::unique_ptr<PatternNode> Right)
      : PatternNode(PN_Binop, Type, false), Op(Op), Left(std::move(Left)),
        Right(std::move(Right)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> BinopStr = {
        {TargetOpcode::G_ADD, "add"},
        {TargetOpcode::G_SUB, "sub"},
        {TargetOpcode::G_MUL, "mul"},
        {TargetOpcode::G_UMULH, "mulhu"},
        {TargetOpcode::G_SMULH, "mulhs"},
        {TargetOpcode::G_UDIV, "udiv"},
        {TargetOpcode::G_SREM, "srem"},
        {TargetOpcode::G_UREM, "urem"},
        {TargetOpcode::G_SDIV, "sdiv"},
        {TargetOpcode::G_SADDSAT, "saddsat"},
        {TargetOpcode::G_UADDSAT, "uaddsat"},
        {TargetOpcode::G_SSUBSAT, "ssubsat"},
        {TargetOpcode::G_USUBSAT, "usubsat"},
        {TargetOpcode::G_SSHLSAT, "sshlsat"},
        {TargetOpcode::G_USHLSAT, "ushlsat"},
        {TargetOpcode::G_SMULFIX, "smulfix"},
        {TargetOpcode::G_UMULFIX, "umulfix"},
        {TargetOpcode::G_SMULFIXSAT, "smulfixsat"},
        {TargetOpcode::G_UMULFIXSAT, "umulfixsat"},
        {TargetOpcode::G_SDIVFIX, "sdivfix"},
        {TargetOpcode::G_UDIVFIX, "udivfix"},
        {TargetOpcode::G_SDIVFIXSAT, "sdivfixsat"},
        {TargetOpcode::G_UDIVFIXSAT, "udivfixsat"},
        {TargetOpcode::G_AND, "and"},
        {TargetOpcode::G_OR, "or"},
        {TargetOpcode::G_XOR, "xor"},
        {TargetOpcode::G_SHL, "shl"},
        {TargetOpcode::G_LSHR, "srl"},
        {TargetOpcode::G_ASHR, "sra"},
        {TargetOpcode::G_SMAX, "smax"},
        {TargetOpcode::G_UMAX, "umax"},
        {TargetOpcode::G_SMIN, "smin"},
        {TargetOpcode::G_UMIN, "umin"},
        {TargetOpcode::G_ROTR, "rotr"},
        {TargetOpcode::G_ROTL, "rotl"},
        {TargetOpcode::G_EXTRACT_VECTOR_ELT, "vector_extract"}};

    static const std::vector<double> CommOps = {TargetOpcode::G_ADD, TargetOpcode::G_MUL, TargetOpcode::G_UMULH, TargetOpcode::G_SMULH, TargetOpcode::G_AND, TargetOpcode::G_OR, TargetOpcode::G_XOR, TargetOpcode::G_UMAX, TargetOpcode::G_SMIN, TargetOpcode::G_UMIN}; // TODO: extend list
    bool IsCommutable = std::find(CommOps.begin(), CommOps.end(), Op) != CommOps.end();
    // RegisterNode* LeftReg = static_cast<RegisterNode*>(Left.get());
    // RegisterNode* RightReg = static_cast<RegisterNode*>(Right.get());
    // bool LeftImm = (LeftReg != 0) ? LeftReg->IsImm : false;
    // bool RightImm = (RightReg != 0) ? RightReg->IsImm : false;
    bool LeftImm = Left->IsImm;
    bool RightImm = Right->IsImm;
    bool DoSwap = IsCommutable && LeftImm && !RightImm;
    std::string TypeStr = lltToString(Type);
    std::string LhsTypeStr = lltToString(Left->Type);
    std::string RhsTypeStr = lltToString(Right->Type);

    // Explicitly specifying types for all ops increases pattern compile time
    // significantly, so we only do for ops where deduction fails otherwise.
    bool PrintType = false;
    bool PrintSrcTypes = false;
    PrintType |= Type.getSizeInBits() != XLen;
    switch (Op) {
    case TargetOpcode::G_SHL:
    case TargetOpcode::G_LSHR:
    case TargetOpcode::G_ASHR:
      PrintType |= true;
      PrintSrcTypes |= true;
      break;
    default:
      break;
    }
    std::string LeftString = (DoSwap ? Right : Left)->patternString(Indent + 1);
    std::string RightString = (DoSwap ? Left : Right)->patternString(Indent + 1);
    if (PrintSrcTypes) {
      LeftString = "(" + (DoSwap ? RhsTypeStr : LhsTypeStr) + " " + LeftString + ")";
      RightString = "(" + (DoSwap ? LhsTypeStr : RhsTypeStr) + " " + RightString + ")";
    }
    std::string OpString = "(" + std::string(BinopStr.at(Op)) + " " +
                           LeftString + ", " + RightString + ")";

    if (PrintType)
      return "(" + TypeStr + " " + OpString + ")";
    return OpString;
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;

    auto LeftT = Left->getRegisterTy(OperandId);
    return LeftT.isValid() ? LeftT : Right->getRegisterTy(OperandId);
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Binop;
  }
};

struct CompareNode : public BinopNode {
  CmpInst::Predicate Cond;

  CompareNode(LLT Type, CmpInst::Predicate Cond,
              std::unique_ptr<PatternNode> Left,
              std::unique_ptr<PatternNode> Right)
      : BinopNode(Type, ISD::SETCC, std::move(Left), std::move(Right)),
        Cond(Cond) {}

  std::string patternString(int Indent = 0) override {
    std::string TypeStr = lltToString(Type);
    std::string LhsTypeStr = lltToString(Left->Type);
    std::string RhsTypeStr = lltToString(Right->Type);

    return "(" + TypeStr + " (setcc (" + LhsTypeStr + " "+ Left->patternString(Indent + 1) + "), (" + RhsTypeStr + " " +
           Right->patternString(Indent + 1) + "), " + CmpStr.at(Cond) + "))";
  }
};

struct SelectNode : public PatternNode {
  ISD::CondCode Cond;
  std::unique_ptr<PatternNode> Left;
  std::unique_ptr<PatternNode> Right;
  std::unique_ptr<PatternNode> Tval;
  std::unique_ptr<PatternNode> Fval;

  SelectNode(LLT Type, ISD::CondCode Cond, std::unique_ptr<PatternNode> Left,
             std::unique_ptr<PatternNode> Right,
             std::unique_ptr<PatternNode> Tval,
             std::unique_ptr<PatternNode> Fval)
      : PatternNode(PN_Select, Type, false), Cond(Cond), Left(std::move(Left)),
        Right(std::move(Right)), Tval(std::move(Tval)), Fval(std::move(Fval)) {}

  std::string patternString(int Indent = 0) override {
    std::string TypeStr = lltToString(Type);

    return "(" + TypeStr + " (riscv_selectcc " +
           Left->patternString(Indent + 1) + ", " +
           Right->patternString(Indent + 1) + ", " + CmpStr.at(Cond) + ", " +
           Tval->patternString(Indent + 1) + ", " +
           Fval->patternString(Indent + 1) + "))";
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;

    for (auto *Operand : {&Left, &Right, &Tval, &Fval}) {
      auto T = (*Operand)->getRegisterTy(OperandId);
      if (T.isValid())
        return T;
    }
    return LLT();
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Select;
  }
};

struct UnopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Operand;

  UnopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Operand)
      : PatternNode(PN_Unop, Type, false), Op(Op), Operand(std::move(Operand)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> UnopStr = {
        {TargetOpcode::G_ANYEXT, "anyext"},
        {TargetOpcode::G_SEXT, "sext"},
        {TargetOpcode::G_ZEXT, "zext"},
        {TargetOpcode::G_VECREDUCE_ADD, "vecreduce_add"},
        {TargetOpcode::G_TRUNC, "trunc"},
        {TargetOpcode::G_BSWAP, "bswap"},
        {TargetOpcode::G_BITREVERSE, "bitreverse"},
        {TargetOpcode::G_BITCAST, "bitcast"},
        {TargetOpcode::G_CTLZ, "ctlz"},
        {TargetOpcode::G_CTTZ, "cttz"},
        {TargetOpcode::G_CTLZ_ZERO_UNDEF, "ctlz_zero_undef"},
        {TargetOpcode::G_CTTZ_ZERO_UNDEF, "cttz_zero_undef"},
        {TargetOpcode::G_CTPOP, "ctpop"},
        {TargetOpcode::G_ABS, "abs"}};

    std::string TypeStr = lltToString(Type);

    // ignore bitcast ops for now
    if (Op == TargetOpcode::G_BITCAST)
      return Operand->patternString(Indent);

    return "(" + TypeStr + " (" + std::string(UnopStr.at(Op)) + " " +
           Operand->patternString(Indent + 1) + "))";
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1 && Op != TargetOpcode::G_BITCAST)
      return Type;
    return Operand->getRegisterTy(OperandId);
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Unop;
  }
};

struct ConstantNode : public PatternNode {
  uint64_t Constant;
  ConstantNode(LLT Type, uint64_t Const)
      : PatternNode(PN_Constant, Type, true), Constant(Const) {}

  std::string patternString(int Indent = 0) override {
    std::string ConstantStr = (XLen == 64) ? std::to_string((int64_t)Constant)
                                           : std::to_string((int32_t)Constant);
    if (Type.isFixedVector()) {

      std::string TypeStr = lltToString(Type);
      return "(" + TypeStr + " (" + RegT + " " + ConstantStr + "))";
    }
    return "(" + lltToString(Type) + " " + ConstantStr + ")";
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Constant;
  }
};

struct RegisterNode : public PatternNode {

  bool IsImm;
  StringRef Name;

  int Offset;
  int Size;
  bool Sext;
  bool VectorExtract =
      false; // TODO: set based on type of this register in other uses

  size_t RegIdx;

  RegisterNode(LLT Type, StringRef Name, size_t RegIdx, bool IsImm, int Offset,
               int Size, bool Sext)
      : PatternNode(PN_Register, Type, IsImm), Name(Name),
        Offset(Offset), Size(Size), Sext(Sext), RegIdx(RegIdx) {}

  std::string patternString(int Indent = 0) override {
    std::string TypeStr = lltToString(Type);
    bool PrintType = false;
    // bool PrintType = true;

    if (IsImm) {
      // Immediate Operands
      assert(Offset == 0 && "immediates must have offset 0");
      return ("(" + RegT + " ") + (Sext ? "simm" : "uimm") +
             std::to_string(Size) + ":$" + std::string(Name) + ")";
    }

    // Full-Size Register Operands
    if ((uint64_t)Size == XLen) {
      std::string Str;
      if (Type.isScalar() && Type.getSizeInBits() == XLen)
        Str = "GPR:$" + std::string(Name);
      if (PrintType)
        return "(" + TypeStr + " " + Str + ")";
      return Str;
      abort();
    }

    // Vector Types (currently rv32 only)
    if ((uint64_t)Size == 32 && XLen == 32) {
      std::string Str;
      if (Type.isFixedVector() && Type.getSizeInBits() == 32 &&
          Type.getElementType().isScalar() &&
          Type.getElementType().getSizeInBits() == 8)
        Str = "GPR32V4:$" + std::string(Name);
      if (Type.isFixedVector() && Type.getSizeInBits() == 32 &&
          Type.getElementType().isScalar() &&
          Type.getElementType().getSizeInBits() == 16)
        Str = "GPR32V2:$" + std::string(Name);
      if (PrintType)
        return "(" + TypeStr + " " + Str + ")";
      return Str;
      abort();
    }

    // Sub-Register Operands
    if (Size == 8 || Size == 16 || (Size == 32 && XLen == 64)) {
      std::string Str;
      if (VectorExtract) {
        Str = std::string("(i32 (vector_extract GPR32V") +
              ((Size == 16) ? "2" : "4") + ":$" + std::string(Name) + ", " +
              std::to_string((Size == 16) ? (Offset / 2) : (Offset)) + "))";
      } else {
        // 32-bit is a supported type, so we can cast instead of shift/mask
        if (Offset == 0 && Size == 32)
          Str = "(i32 GPR:$" + std::string(Name) + ")";
        else if (Offset == 0)
          Str = "GPR:$" + std::string(Name);
        else
          Str = ("(" + RegT + " ") + "(srl GPR:$" + std::string(Name) +
                (" (" + RegT + " ") + std::to_string(Offset * 8) + ")))";
      }
      if (PrintType)
        return "(" + TypeStr + " " + Str + ")";
      return Str;
    }

    abort();
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Register;
  }
};

static std::pair<PatternError, std::unique_ptr<PatternNode>>
traverse(MachineRegisterInfo &MRI, MachineInstr &Cur);

static std::pair<PatternError, std::unique_ptr<PatternNode>>
traverseOperand(MachineRegisterInfo &MRI, MachineInstr &Cur, int Start) {
  assert(Cur.getOperand(1).isReg() && "expected register");
  auto *Op = MRI.getOneDef(Cur.getOperand(1).getReg());
  if (!Op)
    return std::make_pair(FORMAT, nullptr);
  auto [Err, Node] = traverse(MRI, *Op->getParent());
  if (Err)
    return std::make_pair(Err, nullptr);

  return std::make_pair(SUCCESS, std::move(Node));
}

static std::tuple<PatternError, std::unique_ptr<PatternNode>,
                  std::unique_ptr<PatternNode>, std::unique_ptr<PatternNode>>
traverseTernopOperands(MachineRegisterInfo &MRI, MachineInstr &Cur,
                       int Start = 1) {
  assert(Cur.getOperand(Start).isReg() && "expected register");
  auto *First = MRI.getOneDef(Cur.getOperand(Start).getReg());
  if (!First)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr,
                           nullptr);
  assert(Cur.getOperand(Start + 1).isReg() && "expected register");
  auto *Second = MRI.getOneDef(Cur.getOperand(Start + 1).getReg());
  if (!Second)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr,
                           nullptr);
  assert(Cur.getOperand(Start + 2).isReg() && "expected register");
  auto *Third = MRI.getOneDef(Cur.getOperand(Start + 2).getReg());
  if (!Third)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr,
                           nullptr);

  auto [ErrFirst, NodeFirst] = traverse(MRI, *First->getParent());
  if (ErrFirst)
    return std::make_tuple(ErrFirst, nullptr, nullptr, nullptr);

  auto [ErrSecond, NodeSecond] = traverse(MRI, *Second->getParent());
  if (ErrSecond)
    return std::make_tuple(ErrSecond, nullptr, nullptr, nullptr);

  auto [ErrThird, NodeThird] = traverse(MRI, *Third->getParent());
  if (ErrThird)
    return std::make_tuple(ErrThird, nullptr, nullptr, nullptr);

  return std::make_tuple(SUCCESS, std::move(NodeFirst), std::move(NodeSecond),
                         std::move(NodeThird));
}

static std::tuple<PatternError, std::unique_ptr<PatternNode>,
                  std::unique_ptr<PatternNode>>
traverseBinopOperands(MachineRegisterInfo &MRI, MachineInstr &Cur,
                      int Start = 1) {
  assert(Cur.getOperand(Start).isReg() && "expected register");
  auto *LHS = MRI.getOneDef(Cur.getOperand(Start).getReg());
  if (!LHS)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr);
  assert(Cur.getOperand(Start + 1).isReg() && "expected register");
  auto *RHS = MRI.getOneDef(Cur.getOperand(Start + 1).getReg());
  if (!RHS)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr);

  auto [ErrL, NodeL] = traverse(MRI, *LHS->getParent());
  if (ErrL)
    return std::make_tuple(ErrL, nullptr, nullptr);

  auto [ErrR, NodeR] = traverse(MRI, *RHS->getParent());
  if (ErrR)
    return std::make_tuple(ErrR, nullptr, nullptr);
  return std::make_tuple(SUCCESS, std::move(NodeL), std::move(NodeR));
}

static std::tuple<PatternError, std::unique_ptr<PatternNode>>
traverseUnopOperands(MachineRegisterInfo &MRI, MachineInstr &Cur,
                     int Start = 1) {
  assert(Cur.getOperand(Start).isReg() && "expected register");
  auto *RHS = MRI.getOneDef(Cur.getOperand(Start).getReg());
  if (!RHS)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr);

  auto [ErrR, NodeR] = traverse(MRI, *RHS->getParent());
  if (ErrR)
    return std::make_tuple(ErrR, nullptr);
  return std::make_tuple(SUCCESS, std::move(NodeR));
}

static std::tuple<PatternError, std::vector<std::unique_ptr<PatternNode>>>
traverseNOpOperands(MachineRegisterInfo &MRI, MachineInstr &Cur, size_t N,
                    int Start = 1) {
  std::vector<std::unique_ptr<PatternNode>> Operands(N);
  for (size_t I = 0; I < N; I++) {
    // llvm::outs() << "i=" << i << '\n';
    assert(Cur.getOperand(Start + I).isReg() && "expected register");
    auto *Node = MRI.getOneDef(Cur.getOperand(Start + I).getReg());
    if (!Node) {
      // llvm::outs() << "Err" << '\n';
      return std::make_tuple(PatternError(FORMAT, &Cur),
                             std::vector<std::unique_ptr<PatternNode>>());
    }

    auto [Err_, Node_] = traverse(MRI, *Node->getParent());
    if (Err_) {
      // llvm::outs() << "Err2" << '\n';
      return std::make_tuple(Err_, std::vector<std::unique_ptr<PatternNode>>());
    }
    // return std::make_tuple(SUCCESS, std::move(NodeR));
    Operands[I] = std::move(Node_);
  }
  return std::make_tuple(SUCCESS, std::move(Operands));
}

static int getArgIdx(MachineRegisterInfo &MRI, Register Reg) {
  auto It = std::find_if(MRI.livein_begin(), MRI.livein_end(),
                         [&](std::pair<MCRegister, Register> const &E) {
                           return E.first == Reg.asMCReg();
                         });

  if (It == MRI.livein_end())
    return -1;
  return It - MRI.livein_begin();
}

static CDSLInstr::Field const *getArgField(MachineRegisterInfo &MRI,
                                           Register Reg) {
  uint Idx = getArgIdx(MRI, Reg);
  if (Idx > CurInstr->fields.size())
    return nullptr;
  return &CurInstr->fields[Idx];
}

static auto getArgInfo(MachineRegisterInfo &MRI, Register Reg) {
  return std::make_pair(getArgIdx(MRI, Reg), getArgField(MRI, Reg));
}

static std::pair<PatternError, std::unique_ptr<PatternNode>>
traverse(MachineRegisterInfo &MRI, MachineInstr &Cur) {

  switch (Cur.getOpcode()) {
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_UMULH:
  case TargetOpcode::G_SMULH:
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_SADDSAT:
  case TargetOpcode::G_UADDSAT:
  case TargetOpcode::G_SSUBSAT:
  case TargetOpcode::G_USUBSAT:
  case TargetOpcode::G_SSHLSAT:
  case TargetOpcode::G_USHLSAT:
  case TargetOpcode::G_SMULFIX:
  case TargetOpcode::G_UMULFIX:
  case TargetOpcode::G_SMULFIXSAT:
  case TargetOpcode::G_UMULFIXSAT:
  case TargetOpcode::G_SDIVFIX:
  case TargetOpcode::G_UDIVFIX:
  case TargetOpcode::G_SDIVFIXSAT:
  case TargetOpcode::G_UDIVFIXSAT:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SMAX:
  case TargetOpcode::G_UMAX:
  case TargetOpcode::G_SMIN:
  case TargetOpcode::G_UMIN:
  case TargetOpcode::G_EXTRACT_VECTOR_ELT:
  case TargetOpcode::G_ROTR:
  case TargetOpcode::G_ROTL:
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {

    auto [Err, NodeL, NodeR] = traverseBinopOperands(MRI, Cur);
    if (Err)
      return std::make_pair(Err, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    auto Node = std::make_unique<BinopNode>(
        MRI.getType(Cur.getOperand(0).getReg()), Cur.getOpcode(),
        std::move(NodeL), std::move(NodeR));

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_ANYEXT:
  case TargetOpcode::G_SEXT:
  case TargetOpcode::G_ZEXT:
  case TargetOpcode::G_VECREDUCE_ADD:
  case TargetOpcode::G_TRUNC:
  case TargetOpcode::G_BSWAP:
  case TargetOpcode::G_BITREVERSE:
  case TargetOpcode::G_CTLZ:
  case TargetOpcode::G_CTTZ:
  case TargetOpcode::G_CTLZ_ZERO_UNDEF:
  case TargetOpcode::G_CTTZ_ZERO_UNDEF:
  case TargetOpcode::G_CTPOP:
  case TargetOpcode::G_ABS: {

    auto [Err, NodeR] = traverseUnopOperands(MRI, Cur);
    if (Err)
      return std::make_pair(Err, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    auto Node =
        std::make_unique<UnopNode>(MRI.getType(Cur.getOperand(0).getReg()),
                                   Cur.getOpcode(), std::move(NodeR));

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_BITCAST: {
    assert(Cur.getOperand(1).isReg() && "expected register");
    auto *Operand = MRI.getOneDef(Cur.getOperand(1).getReg());
    if (!Operand)
      return std::make_pair(PatternError(FORMAT_LOAD, &Cur), nullptr);

    auto [Err, Node] = traverse(MRI, *Operand->getParent());
    if (Err)
      return std::make_pair(Err, nullptr);

    // if the bitcasted value is a register access, we need to patch the
    // register access type
    if (auto *AsRegNode = llvm::dyn_cast<RegisterNode>(Node.get())) {
      assert(Cur.getOperand(0).isReg() && "expected register");
      AsRegNode->Type = MRI.getType(Cur.getOperand(0).getReg());
      PatternArgs[AsRegNode->RegIdx].ArgTypeStr =
          lltToRegTypeStr(AsRegNode->Type);
    }

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_LOAD: {

    int ReadOffset = 0;
    int ReadSize;

    MachineMemOperand *MMO = *Cur.memoperands_begin();
    ReadSize = MMO->getSizeInBits().getValue();

    assert(Cur.getOperand(1).isReg() && "expected register");
    auto *Addr = MRI.getOneDef(Cur.getOperand(1).getReg());
    if (!Addr)
      return std::make_pair(PatternError(FORMAT_LOAD, &Cur), nullptr);
    auto *AddrI = Addr->getParent();

    if (AddrI->getOpcode() == TargetOpcode::G_PTR_ADD) {
      assert(AddrI->getOperand(1).isReg());
      auto *BaseAddr =
          MRI.getOneDef(AddrI->getOperand(1).getReg())->getParent();
      auto *Offset = MRI.getOneDef(AddrI->getOperand(2).getReg())->getParent();
      AddrI = BaseAddr;

      if (Offset->getOpcode() != TargetOpcode::G_CONSTANT)
        return std::make_pair(PatternError(FORMAT_LOAD, Offset), nullptr);

      ReadOffset = Offset->getOperand(1).getCImm()->getLimitedValue();
    }
    if (AddrI->getOpcode() != TargetOpcode::COPY)
      return std::make_pair(PatternError(FORMAT_LOAD, AddrI), nullptr);

    assert(Cur.getOperand(1).isReg() && "expected register");
    auto AddrLI = AddrI->getOperand(1).getReg();
    if (!MRI.isLiveIn(AddrLI) || !AddrLI.isPhysical())
      return std::make_pair(PatternError(FORMAT_LOAD, AddrI), nullptr);

    auto [Idx, Field] = getArgInfo(MRI, AddrLI);
    if (Field == nullptr)
      return std::make_pair(PatternError(FORMAT_LOAD, AddrI), nullptr);

    PatternArgs[Idx].Llt = MRI.getType(Cur.getOperand(0).getReg());
    PatternArgs[Idx].ArgTypeStr = lltToRegTypeStr(PatternArgs[Idx].Llt);
    PatternArgs[Idx].In = true;

    assert(Cur.getOperand(0).isReg() && "expected register");
    auto Node = std::make_unique<RegisterNode>(
        MRI.getType(Cur.getOperand(0).getReg()), Field->ident, Idx, false,
        ReadOffset, ReadSize, false);

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_CONSTANT: {
    auto *Imm = Cur.getOperand(1).getCImm();
    assert(Cur.getOperand(0).isReg() && "expected register");
    return std::make_pair(SUCCESS, std::make_unique<ConstantNode>(
                                       MRI.getType(Cur.getOperand(0).getReg()),
                                       Imm->getLimitedValue()));
  }
  case TargetOpcode::G_IMPLICIT_DEF: {
    assert(Cur.getOperand(0).isReg() && "expected register");
    return std::make_pair(SUCCESS,
                          std::make_unique<ConstantNode>(
                              MRI.getType(Cur.getOperand(0).getReg()), 0));
  }
  case TargetOpcode::G_ICMP: {
    auto Pred = Cur.getOperand(1);
    auto [Err, NodeL, NodeR] = traverseBinopOperands(MRI, Cur, 2);
    if (Err)
      return std::make_pair(Err, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    return std::make_pair(SUCCESS, std::make_unique<CompareNode>(
                                       MRI.getType(Cur.getOperand(0).getReg()),
                                       (CmpInst::Predicate)Pred.getPredicate(),
                                       std::move(NodeL), std::move(NodeR)));
  }
  case TargetOpcode::COPY: {
    // Immediate Operands
    assert(Cur.getOperand(1).isReg() && "expected register");
    auto Reg = Cur.getOperand(1).getReg();
    auto [Idx, Field] = getArgInfo(MRI, Reg);

    PatternArgs[Idx].In = true;
    PatternArgs[Idx].Llt = LLT();
    PatternArgs[Idx].ArgTypeStr =
        makeImmTypeStr(Field->len, Field->type & CDSLInstr::SIGNED);

    if (Field == nullptr)
      return std::make_pair(FORMAT_IMM, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    return std::make_pair(SUCCESS, std::make_unique<RegisterNode>(
                                       MRI.getType(Cur.getOperand(0).getReg()),
                                       Field->ident, Idx, true, 0, Field->len,
                                       Field->type & CDSLInstr::SIGNED));
  }
  case TargetOpcode::G_BUILD_VECTOR: {
    size_t N = Cur.getNumOperands();
    auto [Err, operands] = traverseNOpOperands(MRI, Cur, N - 1);
    if (Err)
      return std::make_pair(Err, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");

    auto Node =
        std::make_unique<NOpNode>(MRI.getType(Cur.getOperand(0).getReg()),
                                  Cur.getOpcode(), std::move(operands));

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_FSHL:
  case TargetOpcode::G_FSHR:
  case TargetOpcode::G_SELECT:
  case TargetOpcode::G_INSERT_VECTOR_ELT: {
    auto [Err, NodeFirst, NodeSecond, NodeThird] =
        traverseTernopOperands(MRI, Cur);
    if (Err)
      return std::make_pair(Err, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    auto Node = std::make_unique<TernopNode>(
        MRI.getType(Cur.getOperand(0).getReg()), Cur.getOpcode(),
        std::move(NodeFirst), std::move(NodeSecond), std::move(NodeThird));

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_SHUFFLE_VECTOR: {
    assert(Cur.getOperand(1).isReg() && "expected register");
    auto *First = MRI.getOneDef(Cur.getOperand(1).getReg());
    if (!First)
      return std::make_pair(PatternError(FORMAT, &Cur), nullptr);
    assert(Cur.getOperand(2).isReg() && "expected register");
    auto *Second = MRI.getOneDef(Cur.getOperand(2).getReg());
    if (!Second)
      return std::make_pair(PatternError(FORMAT, &Cur), nullptr);
    assert(Cur.getOperand(3).isShuffleMask() && "expected shufflemask");
    ArrayRef<int> Mask = Cur.getOperand(3).getShuffleMask();

    auto [ErrFirst, NodeFirst] = traverse(MRI, *First->getParent());
    if (ErrFirst)
      return std::make_pair(ErrFirst, nullptr);

    auto [ErrSecond, NodeSecond] = traverse(MRI, *Second->getParent());
    if (ErrSecond)
      return std::make_pair(ErrSecond, nullptr);

    assert(Cur.getOperand(0).isReg() && "expected register");
    auto Node = std::make_unique<ShuffleNode>(
        MRI.getType(Cur.getOperand(0).getReg()), Cur.getOpcode(),
        std::move(NodeFirst), std::move(NodeSecond), Mask);

    return std::make_pair(SUCCESS, std::move(Node));
  }
  }

  return std::make_pair(PatternError(FORMAT, &Cur), nullptr);
}

static std::pair<PatternError, std::unique_ptr<PatternNode>>
generatePattern(MachineFunction &MF) {

  if (MF.size() != 1)
    return std::make_pair(MULTIPLE_BLOCKS, nullptr);

  MachineBasicBlock &BB = *MF.begin();
  MachineRegisterInfo &MRI = MF.getRegInfo();

  auto Instrs = BB.instr_rbegin();
  auto InstrsEnd = BB.instr_rend();

  // We expect the pattern block to end with a return immediately preceeded by a
  // store which stores the destination register value.

  if (Instrs == InstrsEnd || !Instrs->isReturn())
    return std::make_pair(FORMAT_RETURN, nullptr);
  Instrs++;
  if (Instrs == InstrsEnd || Instrs->getOpcode() != TargetOpcode::G_STORE)
    return std::make_pair(FORMAT_STORE, nullptr);

  auto &Store = *Instrs;
  MachineMemOperand *MMO = *Store.memoperands_begin();
  if (MMO->getSizeInBits() != XLen && MMO->getSizeInBits() != 32)
    return std::make_pair(FORMAT_STORE, nullptr);

  auto *Addr = MRI.getOneDef(Store.getOperand(1).getReg());
  if (Addr == nullptr || (Addr = MRI.getOneDef(Addr->getReg())) == nullptr ||
      Addr->getParent()->getOpcode() != TargetOpcode::COPY)
    return std::make_pair(FORMAT_STORE, nullptr);

  auto [Idx, Field] =
      getArgInfo(MRI, Addr->getParent()->getOperand(1).getReg());
  PatternArgs[Idx].Out = true;

  auto *RootO = MRI.getOneDef(Store.getOperand(0).getReg());
  if (RootO == nullptr)
    return std::make_pair(FORMAT_STORE, nullptr);
  auto *Root = RootO->getParent();
  {
    LLT Type;
    if (Root->getOpcode() == TargetOpcode::G_BITCAST)
      Type = MRI.getType(Root->getOperand(1).getReg());
    else
      Type = MRI.getType(Root->getOperand(0).getReg());
    PatternArgs[Idx].Llt = Type;
    PatternArgs[Idx].ArgTypeStr = lltToRegTypeStr(Type);
  }

  return traverse(MRI, *Root);
}

bool PatternGen::runOnMachineFunction(MachineFunction &MF) {

  // for convenience
  XLen = PatternGenArgs::Args.Is64Bit ? 64 : 32;
  RegT = PatternGenArgs::Args.Is64Bit ? "i64" : "i32";

  std::string InstName = MF.getName().str().substr(4);
  std::string InstNameO = InstName;
  ++PatternGenNumInstructionsProcessed;
  {
    auto It = std::find_if(
        PatternGenArgs::Instrs->begin(), PatternGenArgs::Instrs->end(),
        [&](CDSLInstr const &Inst) { return Inst.name == InstName; });
    assert(It != PatternGenArgs::Instrs->end() &&
           "implementation function without instruction definition");
    CurInstr = It.base();
  }

  // We use the PatternArgs vector to store additional information
  // about parameters that may be found during pattern gen.
  PatternArgs.clear();
  PatternArgs.append(CurInstr->fields.size(), PatternArg());

  auto [Err, Node] = generatePattern(MF);
  if (Err) {
    llvm::errs() << "Pattern Generation failed for " << MF.getName() << ": "
                 << Errors[Err.Type] << '\n';
    ++(*ErrorStats[Err.Type]);
    if (Err.Inst) {
      llvm::errs() << "Match failure occurred here:\n";
      llvm::errs() << *Err.Inst << "\n";
    }
    ++PatternGenNumInstructionsFailing;
    ++PatternGenNumPatternsFailing;
    return true;
  }

  llvm::outs() << "Pattern for " << InstName << ": " << Node->patternString()
               << '\n';
  ++PatternGenNumPatternsGenerated;

  LLT OutType;
  std::string OutsString;
  std::string InsString;
  for (size_t I = 0; I < CurInstr->fields.size() - 1; I++) {
    if (PatternArgs[I].In) {
      InsString += PatternArgs[I].ArgTypeStr + ":$" +
                   std::string(CurInstr->fields[I].ident) + ", ";
    }
    if (PatternArgs[I].Out) {
      bool IO = PatternArgs[I].In;
      OutsString += PatternArgs[I].ArgTypeStr + ":$" +
                    std::string(CurInstr->fields[I].ident) +
                    (IO ? "_wb, " : ", ");

      assert(!OutType.isValid());
      OutType = PatternArgs[I].Llt;
    }
  }

  InsString = InsString.substr(0, InsString.size() - 2);
  OutsString = OutsString.substr(0, OutsString.size() - 2);

  auto &OutStream = *PatternGenArgs::OutStream;

  OutStream << "let hasSideEffects = 0, mayLoad = " +
                   std::to_string((int)MayLoad) +
                   ", mayStore = " + std::to_string((int)MayStore) +
                   ", "
                   "isCodeGenOnly = 1";

  OutStream << ", Constraints = \"";
  {
    std::string Constr = "";
    for (size_t I = 0; I < CurInstr->fields.size(); I++) {
      auto const &Field = CurInstr->fields[I];
      if (PatternArgs[I].In && PatternArgs[I].Out)
        Constr += "$" + std::string(Field.ident) + " = $" +
                  std::string(Field.ident) + "_wb, ";
    }
    Constr = Constr.substr(0, Constr.size() - 2);
    OutStream << Constr;
  }
  OutStream << "\" in ";
  OutStream << "def " << InstName << "_ : RVInst_" << InstNameO << "<(outs "
            << OutsString << "), (ins " << InsString << ")>;\n";

  std::string PatternStr = Node->patternString();
  std::string Code = "def : Pat<\n\t(";

  Code += lltToString(OutType) + " " + PatternStr + "),\n\t(" + InstName + "_ ";

  Code += InsString;
  Code += ")>;";
  OutStream << "\n" << Code << "\n\n";

  // Delete all instructions to avoid match failures if patterns are not
  // included
  for (auto &MBB : MF)
    MBB.clear();

  return true;
}
