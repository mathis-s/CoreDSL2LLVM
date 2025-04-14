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

#include "llvm/CodeGen/GlobalISel/PatternGen.h"
#include "../../../tools/pattern-gen/lib/InstrInfo.hpp"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
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
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsRISCV.h"
#include "llvm/MC/MCInstrDesc.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/TypeSize.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/Utils/PredicateInfo.h"

// #include "../../llvm/utils/TableGen/Common/GlobalISel/GlobalISelMatchTable.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>

#define DEBUG_TYPE "pattern-gen"

using namespace llvm;

STATISTIC(PatternGenNumInstructionsProcessed, "Processed instructions");
STATISTIC(PatternGenNumInstructionsFailing, "Failing instructions");
STATISTIC(PatternGenNumPatternsGenerated, "Generated patterns");
STATISTIC(PatternGenNumPatternsFailing, "Failing patterns");
STATISTIC(PatternGenNumErrorMultipleBlocks, "Errors of type: MULTIPLE_BLOCKS");
STATISTIC(PatternGenNumErrorFormatReturn, "Errors of type: FORMAT_RETURN");
STATISTIC(PatternGenNumErrorFormatStore, "Errors of type: FORMAT_STORE");
STATISTIC(PatternGenNumErrorFormatLoad, "Errors of type: FORMAT_LOAD");
STATISTIC(PatternGenNumErrorFormatImm, "Errors of type: FORMAT_IMM");
STATISTIC(PatternGenNumErrorFormat, "Errors of type: FORMAT");
STATISTIC(PatternGenNumErrorMultipleStores, "Errors of type: MULTIPLE STORES");

#ifdef LLVM_GISEL_COV_PREFIX
static cl::opt<std::string>
    CoveragePrefix("gisel-coverage-prefix", cl::init(LLVM_GISEL_COV_PREFIX),
                   cl::desc("Record GlobalISel rule coverage files of this "
                            "prefix if instrumentation was generated"));
#else
static const std::string CoveragePrefix;
#endif

std::ostream *PatternGenArgs::OutStream = nullptr;
std::ostream *PatternGenArgs::OutStreamGISelTable = nullptr;
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
static bool IsBranch = 0;

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
  FORMAT,
  MULTIPLE_STORES
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
                        "format",         "multiple stores"};
llvm::Statistic *ErrorStats[] = {nullptr,
                                 &PatternGenNumErrorMultipleBlocks,
                                 &PatternGenNumErrorFormatReturn,
                                 &PatternGenNumErrorFormatStore,
                                 &PatternGenNumErrorFormatLoad,
                                 &PatternGenNumErrorFormatImm,
                                 &PatternGenNumErrorFormat,
                                 &PatternGenNumErrorMultipleStores};

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

static const std::unordered_map<unsigned, std::string> CmpStrGI = {
    {CmpInst::Predicate::ICMP_EQ, "ICMP_EQ"},
    {CmpInst::Predicate::ICMP_NE, "ICMP_NE"},
    {CmpInst::Predicate::ICMP_SLT, "ICMP_SLT"},
    {CmpInst::Predicate::ICMP_SLE, "ICMP_SLE"},
    {CmpInst::Predicate::ICMP_SGT, "ICMP_SGT"},
    {CmpInst::Predicate::ICMP_SGE, "ICMP_SGE"},
    {CmpInst::Predicate::ICMP_ULT, "ICMP_ULT"},
    {CmpInst::Predicate::ICMP_ULE, "ICMP_ULE"},
    {CmpInst::Predicate::ICMP_UGT, "ICMP_UGT"},
    {CmpInst::Predicate::ICMP_UGE, "ICMP_UGE"},
};

std::string lltToString(LLT Llt) {
  if (Llt.isFixedVector())
    return "v" + std::to_string(Llt.getElementCount().getFixedValue()) +
           lltToString(Llt.getElementType());
  if (Llt.isScalar())
    return "i" + std::to_string(Llt.getSizeInBits());
  if (Llt.isPointer())
    return "iPTR";
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
    PN_Load,
    PN_Select,
    PN_Cast,
    PN_Fork,
    PN_ForkOther,
    PN_Store,
    PN_Branch,
    PN_Root
  };

private:
  const PatternNodeKind Kind;

public:
  PatternNodeKind getKind() const { return Kind; }
  LLT Type;
  PatternNode *Parent = nullptr;
  PatternNode(PatternNodeKind Kind, LLT Type) : Kind(Kind), Type(Type) {}

  virtual std::string patternString() = 0;
  virtual LLT getRegisterTy(int OperandId) const {
    if (OperandId == -1)
      return Type;
    return LLT();
  }
  virtual ~PatternNode() {}

  virtual std::vector<std::unique_ptr<PatternNode> *> getOperands() {
    return {};
  };
};

struct RegisterNode : public PatternNode {

  StringRef Name;
  int Size;
  bool Sext;

  size_t RegIdx;

  bool IsImm;

  RegisterNode(LLT Type, StringRef Name, size_t RegIdx, bool IsImm, int Size,
               bool Sext)
      : PatternNode(PN_Register, Type), Name(Name), Size(Size), Sext(Sext),
        RegIdx(RegIdx), IsImm(IsImm) {}

  std::string patternString() override {
    std::string TypeStr = lltToString(Type);
    bool PrintType = Type.isPointer();

    if (IsImm) {
      // Immediate Operands
      return ("(" + RegT + " ") + (Sext ? "simm" : "uimm") +
             std::to_string(Size) + ":$" + std::string(Name) + ")";
    }

    // Vector Types (currently rv32 only)
    if (Type.isFixedVector()) {
      assert((uint64_t)Size == 32 && XLen == 32);
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
    }

    // Full-Size Register Operands
    if (Size == 32 || Size == 64) {
      std::string Str = "GPR:$" + std::string(Name);
      PrintType |= Size == 32 && XLen == 64;
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

struct ConstantNode : public PatternNode {
  uint64_t Constant;
  ConstantNode(LLT Type, uint64_t Const)
      : PatternNode(PN_Constant, Type), Constant(Const) {}

  std::string patternString() override {
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

struct NOpNode : public PatternNode {
  int Op;
  std::vector<std::unique_ptr<PatternNode>> Operands;
  NOpNode(LLT Type, int Op, std::vector<std::unique_ptr<PatternNode>> Operands)
      : PatternNode(PN_NOp, Type), Op(Op), Operands(std::move(Operands)) {
    for (auto &Operand : this->Operands)
      Operand->Parent = this;
  }

  std::string patternString() override {
    static const std::unordered_map<int, std::string> NOpStr = {
        {TargetOpcode::G_BUILD_VECTOR, "build_vector"},
        {TargetOpcode::G_SELECT, "vselect"}};

    std::string S = "(" + std::string(NOpStr.at(Op)) + " ";
    for (auto &Operand : Operands)
      S += Operand->patternString() + ", ";
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
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    std::vector<std::unique_ptr<PatternNode> *> Rv(Operands.size());
    for (size_t i = 0; i < Rv.size(); i++)
      Rv[i] = &Operands[i];
    return Rv;
  }
};

struct ShuffleNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> First;
  std::unique_ptr<PatternNode> Second;
  ArrayRef<int> Mask;

  ShuffleNode(LLT Type, int Op, std::unique_ptr<PatternNode> First,
              std::unique_ptr<PatternNode> Second, ArrayRef<int> Mask)
      : PatternNode(PN_Shuffle, Type), Op(Op), First(std::move(First)),
        Second(std::move(Second)), Mask(std::move(Mask)) {
    this->First->Parent = this;
    this->Second->Parent = this;
  }

  std::string patternString() override {
    std::string TypeStr = lltToString(Type);
    std::string MaskStr = "";

    for (size_t I = 0; I < Mask.size(); I++) {
      if (I != 0) {
        MaskStr += ", ";
      }
      MaskStr += std::to_string(Mask[I]);
    }
    std::string OpString = "(vector_shuffle<" + MaskStr + "> " +
                           First->patternString() + ", " +
                           Second->patternString() + ")";

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

  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&First, &Second};
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
      : PatternNode(PN_Ternop, Type), Op(Op), First(std::move(First)),
        Second(std::move(Second)), Third(std::move(Third)) {
    this->First->Parent = this;
    this->Second->Parent = this;
    this->Third->Parent = this;
  }

  std::string patternString() override {
    static const std::unordered_map<int, std::string> TernopStr = {
        {TargetOpcode::G_FSHL, "fshl"},
        {TargetOpcode::G_FSHR, "fshr"},
        {TargetOpcode::G_INSERT_VECTOR_ELT, "vector_insert"},
        {TargetOpcode::G_SELECT, "select"}};

    std::string TypeStr = lltToString(Type);
    std::string OpString =
        "(" + std::string(TernopStr.at(Op)) + " " + First->patternString() +
        ", " + Second->patternString() + ", " + Third->patternString() + ")";

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

  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&First, &Second, &Third};
  }
};

static const std::unordered_map<int, std::string> BinopStr = {
    {TargetOpcode::G_ADD, "add"},
    {TargetOpcode::G_PTR_ADD, "ptradd"},
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
    {TargetOpcode::G_ICMP, "icmp"},
    {TargetOpcode::G_EXTRACT_VECTOR_ELT, "vector_extract"}};

struct BinopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Left;
  std::unique_ptr<PatternNode> Right;
  bool Commutable;

  BinopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Left,
            std::unique_ptr<PatternNode> Right, bool Commutable = false)
      : PatternNode(PN_Binop, Type), Op(Op), Left(std::move(Left)),
        Right(std::move(Right)), Commutable(Commutable) {
    this->Left->Parent = this;
    this->Right->Parent = this;
  }

  std::string patternString() override {

    auto IsImmNode = [](const PatternNode *Node) {
      if (auto *AsRegNode = llvm::dyn_cast<RegisterNode>(Node))
        return AsRegNode->IsImm;
      if (llvm::isa<ConstantNode>(Node))
        return true;
      return false;
    };

    bool DoSwap =
        Commutable && IsImmNode(Left.get()) && !IsImmNode(Right.get());
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
    case TargetOpcode::G_PTR_ADD:
      PrintType |= true;
      PrintSrcTypes |= true;
      break;
    default:
      break;
    }
    std::string LeftString = (DoSwap ? Right : Left)->patternString();
    std::string RightString = (DoSwap ? Left : Right)->patternString();
    if (PrintSrcTypes) {
      LeftString =
          "(" + (DoSwap ? RhsTypeStr : LhsTypeStr) + " " + LeftString + ")";
      RightString =
          "(" + (DoSwap ? LhsTypeStr : RhsTypeStr) + " " + RightString + ")";
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

  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Left, &Right};
  }
};

struct CompareNode : public BinopNode {
  CmpInst::Predicate Cond;

  CompareNode(LLT Type, CmpInst::Predicate Cond,
              std::unique_ptr<PatternNode> Left,
              std::unique_ptr<PatternNode> Right, bool Commutable)
      : BinopNode(Type, TargetOpcode::G_ICMP, std::move(Left), std::move(Right),
                  Commutable),
        Cond(Cond) {}

  std::string patternString() override {
    std::string TypeStr = lltToString(Type);
    std::string LhsTypeStr = lltToString(Left->Type);
    std::string RhsTypeStr = lltToString(Right->Type);

    return "(" + TypeStr + " (setcc (" + LhsTypeStr + " " +
           Left->patternString() + "), (" + RhsTypeStr + " " +
           Right->patternString() + "), " + CmpStr.at(Cond) + "))";
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
      : PatternNode(PN_Select, Type), Cond(Cond), Left(std::move(Left)),
        Right(std::move(Right)), Tval(std::move(Tval)), Fval(std::move(Fval)) {
    this->Left->Parent = this;
    this->Right->Parent = this;
    this->Tval->Parent = this;
    this->Fval->Parent = this;
  }

  std::string patternString() override {
    std::string TypeStr = lltToString(Type);

    return "(" + TypeStr + " (riscv_selectcc " + Left->patternString() + ", " +
           Right->patternString() + ", " + CmpStr.at(Cond) + ", " +
           Tval->patternString() + ", " + Fval->patternString() + "))";
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

  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Left, &Right, &Tval, &Fval};
  }
};

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

struct UnopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Operand;

  UnopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Operand)
      : PatternNode(PN_Unop, Type), Op(Op), Operand(std::move(Operand)) {
    this->Operand->Parent = this;
  }

  std::string patternString() override {

    std::string TypeStr = lltToString(Type);

    // ignore bitcast ops for now
    if (Op == TargetOpcode::G_BITCAST)
      return Operand->patternString();

    return "(" + TypeStr + " (" + std::string(UnopStr.at(Op)) + " " +
           Operand->patternString() + "))";
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1 && Op != TargetOpcode::G_BITCAST)
      return Type;
    return Operand->getRegisterTy(OperandId);
  }

  static bool classof(const PatternNode *Pat) {
    return Pat->getKind() == PN_Unop;
  }

  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Operand};
  }
};

struct LoadNode : public PatternNode {

  int Size;
  bool Sext;
  std::unique_ptr<PatternNode> Addr;

  LoadNode(int Size, bool Sext, std::unique_ptr<PatternNode> Addr)
      : PatternNode(PN_Load, LLT::scalar(Size)), Size(Size), Sext(Sext),
        Addr(std::move(Addr)) {
    this->Addr->Parent = this;
  }

  std::string patternString() override {
    if ((size_t)Size == XLen)
      return "(" + RegT + " (load " + Addr->patternString() + "))";
    assert((size_t)Size < XLen && "load size > xlen");
    assert(Size >= 8 && "load size < 8");
    assert(Size % 8 == 0 && "load size unaligned");
    // TODO: use AddrRegImm?
    // TODO: how about anyext?
    if (Sext)
      return "(" + RegT + " (sextloadi" + std::to_string(Size) + " " +
             Addr->patternString() + "))";
    return "(" + RegT + " (zextloadi" + std::to_string(Size) + " " +
           Addr->patternString() + "))";
    abort();
  }

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Load; }
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Addr};
  }
};

struct CastNode : public PatternNode {
  std::unique_ptr<PatternNode> Value;

  CastNode(LLT Type, std::unique_ptr<PatternNode> Value)
      : PatternNode(PN_Cast, Type), Value(std::move(Value)) {
    this->Value->Parent = this;
  }

  std::string patternString() override {
    auto LLTString = lltToString(Type);
    return "(" + LLTString + " " + Value->patternString() + ")";
  }

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Cast; }
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Value};
  }
};

struct ForkOtherNode;
struct ForkNode : public PatternNode {
  std::unique_ptr<PatternNode> Value;
  SmallVector<ForkOtherNode *> OtherUses;

  ForkNode(std::unique_ptr<PatternNode> Value)
      : PatternNode(PN_Fork, Value->Type), Value(std::move(Value)) {
    this->Value->Parent = this;
  }

  std::string patternString() override {
    // assert(0 && "Fork node does not support patternString()");
    return "(fork" + Value->patternString() + ")";
  }

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Fork; }
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Value};
  }
};

struct ForkOtherNode : public PatternNode {
  ForkNode *Fork;

  ForkOtherNode(ForkNode *Fork)
      : PatternNode(PN_ForkOther, Fork->Type), Fork(Fork) {}

  std::string patternString() override {
    // assert(0 && "Fork node does not support patternString()");
    return "(forkother " + Fork->patternString() + ")";
  }

  static bool classof(const PatternNode *p) {
    return p->getKind() == PN_ForkOther;
  }
};

struct StoreNode : public PatternNode {
  std::unique_ptr<PatternNode> Value;
  std::unique_ptr<PatternNode> Addr;

  StoreNode(LLT Type, std::unique_ptr<PatternNode> Value,
            std::unique_ptr<PatternNode> Addr)
      : PatternNode(PN_Store, Type), Value(std::move(Value)),
        Addr(std::move(Addr)) {
    this->Value->Parent = this;
    this->Addr->Parent = this;
  }

  std::string patternString() override {

    std::string ValuePat = Value->patternString();
    std::string AddrPat = Addr->patternString();

    if (Type.getSizeInBits() == XLen)
      return "(store (XLenVT " + ValuePat + "), " + AddrPat + ")";
    if (Type.getSizeInBits() == 8)
      return "(truncstorei8 (XLenVT " + ValuePat + "), " + AddrPat + ")";
    if (Type.getSizeInBits() == 16)
      return "(truncstorei16 (XLenVT " + ValuePat + "), " + AddrPat + ")";
    if (Type.getSizeInBits() == 32)
      return "(truncstorei32 (XLenVT " + ValuePat + "), " + AddrPat + ")";
    abort();
  }

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Cast; }
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Value, &Addr};
  }
};

struct BranchNode : public PatternNode {
  std::unique_ptr<PatternNode> Value;
  // std::unique_ptr<PatternNode> Addr;

  BranchNode(LLT Type, std::unique_ptr<PatternNode> Value
             /*std::unique_ptr<PatternNode> Addr*/)
      : PatternNode(PN_Branch, Type), Value(std::move(Value))
  /*Addr(std::move(Addr))*/ {
    this->Value->Parent = this;
    // this->Addr->Parent = this;
  }

  std::string patternString() override {
    return "(branch " + Value->patternString() + ")";
  }

  static bool classof(const PatternNode *p) {
    return p->getKind() == PN_Branch;
  }
  std::vector<std::unique_ptr<PatternNode> *> getOperands() override {
    return {&Value};
  }
};

struct RootNode : public PatternNode {
  std::vector<std::pair<int, std::unique_ptr<PatternNode>>> Stores;

  RootNode(std::vector<std::pair<int, std::unique_ptr<PatternNode>>> &&Stores)
      : PatternNode(PN_Root, LLT()), Stores(std::move(Stores)) {
    for (auto &[OpIdx, Store] : this->Stores)
      Store->Parent = this;
  }

  std::string patternString() override {
    // assert(Stores.size() == 1 &&
    //        "patternString() only supports single-destination instructions.");
    std::string Str = "(";
    for (auto &Store : Stores)
      Str += Store.second->patternString() + " | ";
    Str += ")";
    return Str;
  }
  static bool classof(const PatternNode *p) { return p->getKind() == PN_Root; }
};

struct PatternExtractor {

  using PatternOrError = std::pair<PatternError, std::unique_ptr<PatternNode>>;
  static PatternOrError pError(PatternErrorT Type, MachineInstr *Inst) {
    return std::make_pair(PatternError(Type, Inst), nullptr);
  }
  static PatternOrError PError(PatternError Error) {
    return std::make_pair(Error, nullptr);
  }
  static PatternOrError PError(PatternErrorT Type) {
    return std::make_pair(PatternError(Type), nullptr);
  }
  static PatternOrError PPattern(std::unique_ptr<PatternNode> Pattern) {
    return std::make_pair(PatternError(SUCCESS), std::move(Pattern));
  }

  llvm::DenseMap<MachineInstr *, PatternNode *> Handled{};

  std::tuple<PatternError, std::unique_ptr<PatternNode>,
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

  std::tuple<PatternError, std::unique_ptr<PatternNode>,
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

  std::tuple<PatternError, std::unique_ptr<PatternNode>>
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

  std::tuple<PatternError, std::vector<std::unique_ptr<PatternNode>>>
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
        return std::make_tuple(Err_,
                               std::vector<std::unique_ptr<PatternNode>>());
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

  void changeTypeRecursive(PatternNode *Node, LLT NewType) {
    Node->Type = NewType;
    if (auto *AsBinop = llvm::dyn_cast<BinopNode>(Node)) {
      if (AsBinop->Op == TargetOpcode::G_ADD) {
        AsBinop->Op = TargetOpcode::G_PTR_ADD;
        changeTypeRecursive(AsBinop->Left.get(), NewType);
      }
    }

    if (auto *AsForkOther = llvm::dyn_cast<ForkOtherNode>(Node)) {
      changeTypeRecursive(AsForkOther->Fork->Parent, NewType);
    }
    if (auto *AsFork = llvm::dyn_cast<ForkNode>(Node)) {
      changeTypeRecursive(AsFork->Value.get(), NewType);
      // todo: all other uses
    }
  }

  PatternOrError traverseMemLoad(MachineRegisterInfo &MRI, MachineInstr &Cur,
                                 int ReadSize, MachineInstr *AddrI) {
    MayLoad = 1;
    if (AddrI->getOpcode() == TargetOpcode::G_INTTOPTR) {
      auto *AddrInt = MRI.getOneDef(AddrI->getOperand(1).getReg());
      auto [Err, Node] = traverse(MRI, *AddrInt->getParent());
      if (Err)
        return PError(Err);
      changeTypeRecursive(Node.get(), LLT::pointer(0, XLen));
      bool Sext = Cur.getOpcode() == TargetOpcode::G_SEXTLOAD;
      return PPattern(
          std::make_unique<LoadNode>(ReadSize, Sext, std::move(Node)));
    }
    if (AddrI->getOpcode() == TargetOpcode::G_PTR_ADD) {
      auto [Err, Node] = traverse(MRI, *AddrI);
      if (Err)
        return PError(Err);
      changeTypeRecursive(Node.get(), LLT::pointer(0, XLen));
      bool Sext = Cur.getOpcode() == TargetOpcode::G_SEXTLOAD;
      return PPattern(
          std::make_unique<LoadNode>(ReadSize, Sext, std::move(Node)));
    }
    abort();
  }

  PatternOrError traverseRegLoad(MachineRegisterInfo &MRI, MachineInstr &Cur,
                                 int ReadSize, MachineInstr *AddrI) {

    int ReadOffset = 0;

    if (AddrI->getOpcode() == TargetOpcode::G_PTR_ADD) {
      assert(AddrI->getOperand(1).isReg());
      auto *BaseAddr =
          MRI.getOneDef(AddrI->getOperand(1).getReg())->getParent();
      auto *Offset = MRI.getOneDef(AddrI->getOperand(2).getReg())->getParent();

      if (Offset->getOpcode() != TargetOpcode::G_CONSTANT)
        return traverseMemLoad(MRI, Cur, ReadSize, AddrI);

      AddrI = BaseAddr;
      ReadOffset = Offset->getOperand(1).getCImm()->getLimitedValue();
    }
    if (AddrI->getOpcode() == TargetOpcode::G_SELECT) {
      // TODO: implement this!
      return pError(FORMAT_LOAD, AddrI);
    }
    if (AddrI->getOpcode() != TargetOpcode::COPY)
      return pError(FORMAT_LOAD, AddrI);

    assert(Cur.getOperand(1).isReg() && "expected register");
    auto AddrLI = AddrI->getOperand(1).getReg();
    if (!MRI.isLiveIn(AddrLI) || !AddrLI.isPhysical())
      return pError(FORMAT_LOAD, AddrI);

    auto [Idx, Field] = getArgInfo(MRI, AddrLI);
    if (Field == nullptr)
      return pError(FORMAT_LOAD, AddrI);

    auto Type = MRI.getType(Cur.getOperand(0).getReg());
    PatternArgs[Idx].Llt = Type;
    PatternArgs[Idx].ArgTypeStr = lltToRegTypeStr(PatternArgs[Idx].Llt);
    PatternArgs[Idx].In = true;

    assert(Cur.getOperand(0).isReg() && "expected register");
    std::unique_ptr<PatternNode> Node = std::make_unique<RegisterNode>(
        Type, Field->ident, Idx, false, Type.getSizeInBits(), false);

    bool SizeMismatch = (int)Type.getSizeInBits() != ReadSize;

    if (Cur.getOpcode() == TargetOpcode::G_ZEXTLOAD && SizeMismatch) {
      if (ReadOffset != 0)
        Node = std::make_unique<BinopNode>(
            Type, TargetOpcode::G_LSHR, std::move(Node),
            std::make_unique<ConstantNode>(Type, ReadOffset * 8));
      if ((uint64_t)(ReadSize + ReadOffset * 8) < XLen) {
        Node = std::make_unique<BinopNode>(
            Type, TargetOpcode::G_AND, std::move(Node),
            std::make_unique<ConstantNode>(Type, (1UL << ReadSize) - 1));
      }
    } else if (Cur.getOpcode() == TargetOpcode::G_SEXTLOAD && SizeMismatch) {
      int Shamt = XLen - ReadSize - ReadOffset * 8;
      auto Left = Shamt == 0 ? std::move(Node)
                             : std::make_unique<BinopNode>(
                                   Type, TargetOpcode::G_SHL, std::move(Node),
                                   std::make_unique<ConstantNode>(Type, Shamt));

      Node = std::make_unique<BinopNode>(
          Type, TargetOpcode::G_ASHR, std::move(Left),
          std::make_unique<ConstantNode>(Type, XLen - ReadSize));
    }

    return PPattern(std::move(Node));
  }

  PatternOrError traverse_impl(MachineRegisterInfo &MRI, MachineInstr &Cur) {

    switch (Cur.getOpcode()) {
    case TargetOpcode::G_ADD:
    case TargetOpcode::G_PTR_ADD:
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
          std::move(NodeL), std::move(NodeR), Cur.isCommutable());

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
    case TargetOpcode::G_LOAD:
    case TargetOpcode::G_ZEXTLOAD:
    case TargetOpcode::G_SEXTLOAD: {

      MachineMemOperand *MMO = *Cur.memoperands_begin();
      int ReadSize = MMO->getSizeInBits().getValue();

      assert(Cur.getOperand(1).isReg() && "expected register");
      auto *Addr = MRI.getOneDef(Cur.getOperand(1).getReg());
      if (!Addr)
        return std::make_pair(PatternError(FORMAT_LOAD, &Cur), nullptr);
      auto *AddrI = Addr->getParent();

      if (AddrI->getOpcode() == TargetOpcode::G_INTTOPTR ||
          AddrI->getOpcode() == TargetOpcode::G_PTR_ADD)
        return traverseMemLoad(MRI, Cur, ReadSize, AddrI);
      return traverseRegLoad(MRI, Cur, ReadSize, AddrI);
    }
    case TargetOpcode::G_CONSTANT: {
      auto *Imm = Cur.getOperand(1).getCImm();
      assert(Cur.getOperand(0).isReg() && "expected register");
      return std::make_pair(
          SUCCESS,
          std::make_unique<ConstantNode>(
              MRI.getType(Cur.getOperand(0).getReg()), Imm->getLimitedValue()));
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
      return std::make_pair(SUCCESS,
                            std::make_unique<CompareNode>(
                                MRI.getType(Cur.getOperand(0).getReg()),
                                (CmpInst::Predicate)Pred.getPredicate(),
                                std::move(NodeL), std::move(NodeR),
                                Cur.isCommutable() ||
                                    Pred.getPredicate() == CmpInst::ICMP_EQ ||
                                    Pred.getPredicate() == CmpInst::ICMP_NE));
    }
    case TargetOpcode::COPY: {
      // Immediate Operands
      assert(Cur.getOperand(1).isReg() && "expected register");
      auto Reg = Cur.getOperand(1).getReg();

      // Copying from a physical reg means this is a function argument,
      // so a register or immediate value in the behavior function.
      if (Reg.isPhysical()) {
        auto [Idx, Field] = getArgInfo(MRI, Reg);

        PatternArgs[Idx].In = true;
        PatternArgs[Idx].Llt = LLT();
        PatternArgs[Idx].ArgTypeStr =
            makeImmTypeStr(Field->len, Field->type & CDSLInstr::SIGNED);

        if (Field == nullptr)
          return std::make_pair(FORMAT_IMM, nullptr);

        assert(Cur.getOperand(0).isReg() && "expected register");
        return std::make_pair(SUCCESS,
                              std::make_unique<RegisterNode>(
                                  MRI.getType(Cur.getOperand(0).getReg()),
                                  Field->ident, Idx, true, Field->len,
                                  Field->type & CDSLInstr::SIGNED));
      }

      // Else COPY is just a pass-through.
      auto [Err, Node] = traverseUnopOperands(MRI, Cur);
      return std::make_pair(Err, std::move(Node));
    }
    case TargetOpcode::G_INTTOPTR: {
      auto [Err, Node] = traverseUnopOperands(MRI, Cur);
      if (Err)
        return PError(Err);

      return PPattern(
          std::make_unique<CastNode>(LLT::pointer(0, XLen), std::move(Node)));
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
      // case TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS: {
      //   auto &asIntr = llvm::cast<GIntrinsic>(Cur);
      //   switch (asIntr.getIntrinsicID()) {
      //   case llvm::Intrinsic::riscv_pg_branch: {
      //     auto *Cond = MRI.getOneDef(asIntr.getOperand(2).getReg());
      //     auto [CondErr, CondV] = traverse(MRI, *Cond->getParent());
      //     if (CondErr)
      //       return PError(CondErr);
      //     return PPattern(std::make_unique<BranchNode>(LLT::scalar(XLen),
      //     std::move(CondV)));
      //   }
      //   default:
      //     llvm_unreachable("unknown intrinsic id");
      //   }
      //   break;
      // }
    }

    return std::make_pair(PatternError(FORMAT, &Cur), nullptr);
  }

  PatternOrError traverse(MachineRegisterInfo &MRI, MachineInstr &Cur) {

    if (Cur.getOpcode() == TargetOpcode::G_CONSTANT)
      return traverse_impl(MRI, Cur);

    if (auto Iter = Handled.find(&Cur); Iter != Handled.end()) {
      // If the value we're looking at has been processed before, we insert a
      // Fork+ForkOther pair to re-use the existing value without duplication.
      // This is required for multi-output.

      // todo: add to existing fork.
      PatternNode *Node = Iter->second;
      PatternNode *Parent = Node->Parent;

      auto Operands = Parent->getOperands();
      auto OperandIter = std::find_if(
          Operands.begin(), Operands.end(),
          [=](std::unique_ptr<PatternNode> *Op) { return Op->get() == Node; });
      assert(OperandIter != Operands.end());
      std::unique_ptr<PatternNode> *Ptr = *OperandIter;

      auto NodeOwning = std::move(*Ptr);
      auto ForkNodeOwning = std::make_unique<ForkNode>(std::move(NodeOwning));
      auto &ForkNode = *ForkNodeOwning;
      (*Ptr) = std::move(ForkNodeOwning);
      ForkNode.Parent = Parent;

      auto RetNode = std::make_unique<ForkOtherNode>(&ForkNode);
      ForkNode.OtherUses.push_back(RetNode.get());
      return PPattern(std::move(RetNode));
    }

    auto Rv = traverse_impl(MRI, Cur);
    if (!Rv.first)
      Handled[&Cur] = Rv.second.get();
    return Rv;
  }

  PatternOrError traverseRegStore(size_t Idx, MachineRegisterInfo &MRI,
                                  MachineInstr &Root) {
    LLT Type;
    if (Root.getOpcode() == TargetOpcode::G_BITCAST)
      Type = MRI.getType(Root.getOperand(1).getReg());
    else
      Type = MRI.getType(Root.getOperand(0).getReg());

    PatternArgs[Idx].Out = true;
    PatternArgs[Idx].Llt = Type;
    PatternArgs[Idx].ArgTypeStr = lltToRegTypeStr(Type);

    return traverse(MRI, Root);
  }

  PatternOrError traverseMemStore(LLT Type, MachineRegisterInfo &MRI,
                                  MachineInstr &Value, MachineInstr &Addr) {
    auto ValueP = traverse(MRI, Value);
    if (ValueP.first)
      return PError(ValueP.first);
    auto AddrP = traverse(MRI, Addr);
    if (AddrP.first)
      return PError(AddrP.first);

    MayStore = 1;

    return PPattern(std::make_unique<StoreNode>(Type, std::move(ValueP.second),
                                                std::move(AddrP.second)));
  }

  PatternOrError traverseStore(MachineRegisterInfo &MRI, MachineInstr &Store,
                               int &OutOpIdx) {
    OutOpIdx = -1;
    MachineMemOperand *MMO = *Store.memoperands_begin();

    auto *ValueR = MRI.getOneDef(Store.getOperand(0).getReg());
    if (ValueR == nullptr)
      return pError(FORMAT_STORE, &Store);
    auto *ValueI = ValueR->getParent();

    auto *Addr = MRI.getOneDef(Store.getOperand(1).getReg());
    if (Addr == nullptr)
      return pError(FORMAT_STORE, &Store);

    auto *AddrD = MRI.getOneDef(Addr->getReg());
    if (AddrD == nullptr)
      return pError(FORMAT_STORE, &Store);

    auto *AddrI = MRI.getOneDef(AddrD->getReg())->getParent();
    if (AddrI->getOpcode() == TargetOpcode::COPY) {
      auto Idx = getArgIdx(MRI, AddrI->getOperand(1).getReg());
      if (Idx != -1) {
        OutOpIdx = Idx;
        if (MMO->getSizeInBits() != XLen && MMO->getSizeInBits() != 32)
          return pError(FORMAT_STORE, &Store);
        return traverseRegStore(Idx, MRI, *ValueI);
      }
    }

    return traverseMemStore(MMO->getType(), MRI, *ValueI, *AddrI);
  }

  PatternOrError generatePattern(MachineFunction &MF) {

    if (MF.size() != 1)
      return std::make_pair(MULTIPLE_BLOCKS, nullptr);

    MachineBasicBlock &BB = *MF.begin();
    MachineRegisterInfo &MRI = MF.getRegInfo();

    auto Instrs = BB.instr_rbegin();
    auto InstrsEnd = BB.instr_rend();

    // We expect the pattern block to end with a return immediately preceeded by
    // a store which stores the destination register value.
    if (Instrs == InstrsEnd || !Instrs->isReturn())
      return PError(FORMAT_STORE);
    Instrs++;

    std::vector<std::pair<int, std::unique_ptr<PatternNode>>> Stores;

    for (; Instrs != InstrsEnd; Instrs++) {
      if (Instrs->getOpcode() == TargetOpcode::G_STORE) {
        int OpIdx;
        auto Result = traverseStore(MRI, *Instrs, OpIdx);
        // Return on error
        if (Result.first)
          return Result;

        Stores.push_back(std::make_pair(OpIdx, std::move(Result.second)));
      }
      if (Instrs->getOpcode() == TargetOpcode::G_INTRINSIC_W_SIDE_EFFECTS) {
        auto &asIntr = llvm::cast<GIntrinsic>(*Instrs);
        switch (asIntr.getIntrinsicID()) {
        case llvm::Intrinsic::riscv_pg_branch: {
          auto *Cond = MRI.getOneDef(asIntr.getOperand(2).getReg());
          auto [CondErr, CondV] = traverse(MRI, *Cond->getParent());
          if (CondErr)
            return PError(CondErr);
          IsBranch = 1;
          // insert an icmp if there isn't one already
          if (!(llvm::isa<BinopNode>(CondV) &&
                llvm::cast<BinopNode>(*CondV).Op == TargetOpcode::G_ICMP)) {
            CondV = std::make_unique<CompareNode>(
                LLT::scalar(XLen), CmpInst::ICMP_NE,
                std::make_unique<ConstantNode>(LLT::scalar(XLen), 0),
                std::move(CondV), true);
          }

          Stores.push_back(
              std::make_pair(0, std::make_unique<BranchNode>(
                                    LLT::scalar(XLen), std::move(CondV))));
          break;
        }
        default:
          llvm_unreachable("unknown intrinsic id");
        }
      }
    }

    return PPattern(std::make_unique<RootNode>(std::move(Stores)));
  }
};

static const char *FixedInstrs[] = {
#undef HANDLE_TARGET_OPCODE
#undef HANDLE_TARGET_OPCODE_MARKER
#define HANDLE_TARGET_OPCODE(OPC) #OPC,
#include "llvm/Support/TargetOpcodes.def"
};
constexpr unsigned NumFixedInstructions = std::size(FixedInstrs) - 1;

class GISelTableBackend {
private:
  std::stringstream Output;

  enum State {
    None,
    None_Ptr,
    OpMatcher,
    InsnMatcher,
  };

  struct BindOutput {
    StringRef Name;
    int SrcInstIdx;
    int SrcInstOpIdx;
    int OutInstOpIdx;
  };

  // Operand indicies may be swapped for commutative ops. This struct is a
  // wrapper containing either a fixed OperandIdx (non-commutative), or a
  // variable OperandIdx referencing a Range.
  struct OperandIdx {
    bool IsRange = 0;
    uint16_t RangeIdx = 0;
    uint16_t RangeLen = 0;
    uint16_t RangeOffs = 0;
    uint16_t OpIdx = 0;

    static OperandIdx none() { return OperandIdx{}; }
    static OperandIdx fixed(int Idx) {
      return OperandIdx{false, 0, 0, 0, (uint16_t)Idx};
    }
    static OperandIdx ranged(size_t RangeIdx, size_t RangeLen, int Offset,
                             int Idx) {
      return OperandIdx{true, (uint16_t)RangeIdx, (uint16_t)RangeLen,
                        (uint16_t)Offset, (uint16_t)Idx};
    }

    auto str() const {
      std::stringstream os;
      if (IsRange)
        os << "(" << RangeOffs << "+(R_" << RangeIdx << "+" << OpIdx << ")%"
           << RangeLen << ")";
      else
        os << OpIdx;
      return os.str();
    }

    friend std::ostream &operator<<(std::ostream &os, const OperandIdx &OpIdx) {
      os << OpIdx.str();
      return os;
    }
  };

  std::vector<BindOutput> BindOutputs = {};
  std::vector<int> MarkErase = {};
  std::vector<int> Ranges = {};
  DenseSet<PatternNode *> CoveredNodes;

  // store the (insnID, operandIdx) referencing each forkOther
  DenseMap<ForkOtherNode *, std::pair<uint32_t, OperandIdx>> ForkOtherOrigOp;

  // store pre-existing ranged operand idx IDs
  DenseMap<PatternNode *, uint32_t> RangedOperandIdxID;

  size_t NumCheckSafeToMove = 0;
  RootNode *Root;
  size_t CurIdx = 1;

  std::optional<std::string> Error = std::nullopt;

public:
  size_t process(RootNode *Root) {
    this->Root = Root;
    assert(Root->Stores.size() >= 1);

    // RootIsCovered.clear();
    // RootIsCovered.reserve(Root->Stores.size());
    // RootIsCovered[0] = true;

    auto &[OpIdx, Store] = Root->Stores[0];

    // If the root node is a PTR_ADD it gets converted to a regular add by a
    // pre-select hook. Not the case for non-root nodes.
    bool PtrAddFixup = false;
    if (auto *AsBinop = llvm::dyn_cast<BinopNode>(Store.get());
        AsBinop && AsBinop->Op == TargetOpcode::G_PTR_ADD) {
      AsBinop->Op = TargetOpcode::G_ADD;
      PtrAddFixup = true;
    }

    if (!llvm::isa<BranchNode>(Store))
      BindOutputs.push_back(
          BindOutput{CurInstr->fields[OpIdx].ident, 0, 0, OpIdx});
    process_impl(Store.get(), 0, InsnMatcher);

    if (PtrAddFixup)
      llvm::cast<BinopNode>(Store.get())->Op = TargetOpcode::G_PTR_ADD;

    // this is returned as a rough measure for complexity
    return CurIdx;
  }
  GISelTableBackend() {
    Output << "{\nRuleMatcher "
              "RM{Locs};\nRuleMatcherScores[RM.getRuleID()] = "
              "16;\nRM.addRequiredFeature(RK.getDef(\"HasVendorXCValu\"));\n"
              "RM.addRequiredFeature(RK.getDef(\"IsRV"
           << XLen
           << "\"));\nInstructionMatcher &_0 = RM.addInstructionMatcher(\"\""
              ");\n";
  }

  std::optional<std::string> getError() { return Error; }

  std::stringstream getOutput() {

    if (BindOutputs.size() +
            (llvm::isa<BranchNode>(Root->Stores[0].second.get()) ? 1 : 0) !=
        Root->Stores.size())
      Error = "could not cover entire pattern. Do all results use at least one "
              "common value?";

    if (Error)
      return std::stringstream{};
    finalize();

    std::stringstream Header;
    for (size_t I = 0; I < Ranges.size(); I++)
      Header << "for (int R_" << I << " = 0; R_" << I << " < " << Ranges[I]
             << "; R_" << I << "++)\n";
    Header << Output.rdbuf();

    return Header;
  }

private:
  void process_impl(PatternNode *Node, size_t Idx = 0,
                    State state = InsnMatcher,
                    OperandIdx OpIdx = OperandIdx::none()) {

    auto AddPredicate = [&](std::string Predicate, ArrayRef<StringRef> Args) {
      Output << "_" << Idx << ".addPredicate<" << Predicate << ">(";
      if (Args.size() != 0) {
        for (auto &Arg : Args)
          Output << Arg.str() << ", ";
        Output.seekp(-2, std::ios_base::end);
      }
      Output << ");\n";
    };

    auto PromoteToOperandMatcher = [&]() {
      if (state >= OpMatcher)
        return;
      Output << "auto &_" << (CurIdx) << " = _" << Idx << ".addOperand("
             << OpIdx << ", \"\", 0);\n";
      Idx = CurIdx++;
      if (state == None_Ptr)
        AddPredicate("PointerToAnyOperandMatcher", {"0"});
      state = OpMatcher;
    };

    auto BindAndPromoteToOperandMatcher = [&](StringRef Name) {
      assert(state == None || state == None_Ptr);
      Output << "auto &_" << (CurIdx) << " = _" << Idx << ".addOperand("
             << OpIdx << ", \"" << Name.str() << "\", 0);\n";
      Idx = CurIdx++;
      if (state == None_Ptr)
        AddPredicate("PointerToAnyOperandMatcher", {"0"});
      state = OpMatcher;
    };

    auto PromoteToInsnMatcher = [&]() {
      PromoteToOperandMatcher();
      if (state >= InsnMatcher)
        return;
      NumCheckSafeToMove++;
      Output << "auto &_" << (CurIdx) << " = (**(_" << Idx
             << ".addPredicate<InstructionOperandMatcher>(RM, "
                "\"\"))).getInsnMatcher();\n";
      Idx = CurIdx++;
      state = InsnMatcher;
    };

    auto PromoteToOtherUseInsnMatcher = [&](OperandIdx ExpectedOperand) {
      PromoteToOperandMatcher();
      if (state >= InsnMatcher)
        return;
      Output << "auto &_" << (CurIdx) << " = (**(_" << Idx
             << ".addPredicate<OtherUseInstructionOperandMatcher>("
             << ExpectedOperand
             << ", RM, "
                "\"\"))).getInsnMatcher();\n";
      Idx = CurIdx++;
      state = InsnMatcher;
    };

    auto CheckOpcode = [&](std::string const &Opcode) {
      assert(state == InsnMatcher);
      Output << "_" << Idx
             << ".addPredicate<InstructionOpcodeMatcher>(&Target."
                "getInstruction(RK.getDef(\""
             << Opcode << "\")));\n";
    };

    auto CheckConstantInt = [&](int64_t Value) {
      assert(state == OpMatcher);
      Output << "_" << Idx << ".addPredicate<ConstantIntOperandMatcher>("
             << Value << ");\n";
    };

    auto CheckIsLLT = [&](LLT Type) {
      assert(state == OpMatcher);
      assert(Type.isScalar() || Type.isPointer()); // todo: other types

      std::string TypeStr;
      if (Type.isScalar())
        TypeStr = "LLT::scalar(" + std::to_string(Type.getSizeInBits()) + ")";
      else if (Type.isPointer())
        TypeStr = "LLT::pointer(0, " + std::to_string(XLen) + ")";

      Output << "_" << Idx << ".addPredicate<LLTOperandMatcher>(" << TypeStr
             << ");\n";
    };

    auto CheckIsRegBank = [&](std::string RegBank) {
      assert(state == OpMatcher);
      Output << "_" << Idx << ".addPredicate<RegisterBankOperandMatcher>("
             << RegBank << ");\n";
    };

    auto RootStoreIdx = [&](PatternNode *Node) -> std::optional<int> {
      auto InstrOpIdxIter = std::find_if(
          Root->Stores.begin(), Root->Stores.end(),
          [&](const auto &Pair) { return Pair.second.get() == Node; });
      if (InstrOpIdxIter != Root->Stores.end())
        return InstrOpIdxIter->first;
      return std::nullopt;
    };

    auto GetOrMakeRangeID = [&](PatternNode *Node) {
      size_t RangeID;
      if (RangedOperandIdxID.contains(Node)) {
        RangeID = RangedOperandIdxID[Node];
      } else {
        RangeID = Ranges.size();
        Ranges.push_back(2);
        RangedOperandIdxID[Node] = RangeID;
      }
      return RangeID;
    };

    auto GetRootOfSubtree = [&](PatternNode *Node) {
      while (1) {
        if (auto Idx = RootStoreIdx(Node))
          return std::make_pair(Node, *Idx);
        Node = Node->Parent;
      }
    };

    auto Downwards = [&](PatternNode *Node) {
      while (1) {
        // We're done if in Root.Stores
        if (RootStoreIdx(Node))
          return;

        switch (Node->getKind()) {
        case PatternNode::PN_Binop:
          CheckOpcode(FixedInstrs[llvm::cast<BinopNode>(Node)->Op]);
          break;
        case PatternNode::PN_Load: {
          auto &AsLoad = llvm::cast<LoadNode>(*Node);
          if (AsLoad.Size == (int)XLen)
            CheckOpcode("G_LOAD");
          else if (AsLoad.Sext)
            CheckOpcode("G_SEXTLOAD");
          else
            CheckOpcode("G_ZEXTLOAD");
          break;
        }
        case PatternNode::PN_Cast:
          Node = Node->Parent;
          continue;
        default:
          abort();
        }
        NumCheckSafeToMove++;

        // Find Operand Index in parent
        auto Operands = Node->Parent->getOperands();
        auto Iter = std::find_if(Operands.begin(), Operands.end(),
                                 [&](std::unique_ptr<PatternNode> *Op) {
                                   return Op->get() == Node;
                                 });
        assert(Iter != Operands.end());
        OperandIdx ExpectedOpIdx =
            OperandIdx::fixed(Iter - Operands.begin() + 1);
        if (auto *AsBinop = llvm::dyn_cast<BinopNode>(Node->Parent);
            AsBinop && AsBinop->Commutable) {
          ExpectedOpIdx = OperandIdx::ranged(GetOrMakeRangeID(AsBinop), 2, 1,
                                             Iter - Operands.begin());
        }
        state = None;
        OpIdx = OperandIdx::fixed(0);
        PromoteToOperandMatcher();
        PromoteToOtherUseInsnMatcher(ExpectedOpIdx);
        Node = Node->Parent;
      }
    };

    CoveredNodes.insert(Node);

    switch (Node->getKind()) {
    case PatternNode::PN_Binop: {
      // bool AsPtr = state == None_Ptr;

      auto &AsBinop = llvm::cast<BinopNode>(*Node);
      PromoteToInsnMatcher();

      assert((unsigned)AsBinop.Op < NumFixedInstructions);
      auto Opcode = AsBinop.Op;
      auto Commutable = AsBinop.Commutable;

      // If the output type is a ptr convert G_ADD to G_PTR_ADD.
      // if (AsPtr && Opcode == TargetOpcode::G_ADD)
      //  Opcode = TargetOpcode::G_PTR_ADD;
      CheckOpcode(FixedInstrs[Opcode]);

      int OpsBase = 1;
      if (Opcode == TargetOpcode::G_ICMP) {
        AddPredicate(
            "CmpPredicateOperandMatcher",
            {"1", "\"" + CmpStrGI.at(static_cast<CompareNode &>(AsBinop).Cond) +
                      "\""});
        OpsBase = 2;
      }

      bool LeftAsPtr = Opcode == TargetOpcode::G_PTR_ADD;

      auto *Left = AsBinop.Left.get();
      auto *Right = AsBinop.Right.get();

      size_t RangeID;
      if (Commutable)
        RangeID = GetOrMakeRangeID(Node);

      process_impl(Left, Idx, LeftAsPtr ? None_Ptr : None,
                   Commutable ? OperandIdx::ranged(RangeID, 2, OpsBase, 0)
                              : OperandIdx::fixed(OpsBase + 0));
      process_impl(Right, Idx, None,
                   Commutable ? OperandIdx::ranged(RangeID, 2, OpsBase, 1)
                              : OperandIdx::fixed(OpsBase + 1));
      break;
    }
    case PatternNode::PN_Register: {
      auto &AsRegNode = llvm::cast<RegisterNode>(*Node);
      bool Ptr = (state == None_Ptr);
      BindAndPromoteToOperandMatcher(AsRegNode.Name);
      if (!Ptr)
        CheckIsLLT(AsRegNode.Type);
      CheckIsRegBank("GPR");
      break;
    }
    case PatternNode::PN_Fork: {
      auto &AsFork = llvm::cast<ForkNode>(*Node);
      assert(AsFork.OtherUses.size() == 1);
      auto &OtherUse = *AsFork.OtherUses[0];

      if (auto StoreIdx = RootStoreIdx(&OtherUse)) {
        assert(state == None || state == None_Ptr);
        auto InsnMatcherIdx = CurIdx + 1;
        process_impl(AsFork.Value.get(), Idx, state, OpIdx);
        BindOutputs.push_back(BindOutput{CurInstr->fields[*StoreIdx].ident,
                                         (int)InsnMatcherIdx, 0, *StoreIdx});
        break;
      }

      auto [OtherRoot, StoreIdx] = GetRootOfSubtree(OtherUse.Parent);
      if (CoveredNodes.contains(OtherRoot)) {
        if (!CoveredNodes.contains(AsFork.Value.get()))
          Error = "bad fork order";
        else {
          auto [OrigIdx, OrigOpIdx] = ForkOtherOrigOp[&OtherUse];

          PromoteToOperandMatcher();
          AddPredicate("SameOperandMatcherByIdx",
                       {"_" + std::to_string(OrigIdx) + ".getInsnVarID()",
                        OrigOpIdx.str(), "0"});
        }

        break;
      }

      auto Operands = OtherUse.Parent->getOperands();
      auto Iter = std::find_if(Operands.begin(), Operands.end(),
                               [&](std::unique_ptr<PatternNode> *Op) {
                                 return Op->get() == &OtherUse;
                               });
      assert(Iter != Operands.end());

      OperandIdx ExpectedOpIdx = OperandIdx::fixed(Iter - Operands.begin() + 1);
      if (auto *AsBinop = llvm::dyn_cast<BinopNode>(OtherUse.Parent);
          AsBinop && AsBinop->Commutable) {

        auto RangeID = GetOrMakeRangeID(AsBinop);
        ExpectedOpIdx =
            OperandIdx::ranged(RangeID, 2, 1, Iter - Operands.begin());
      }
      PromoteToOtherUseInsnMatcher(ExpectedOpIdx);

      // Generate code to search downward until we arrive at Root.
      Downwards(OtherUse.Parent);

      process_impl(OtherRoot, Idx, InsnMatcher);
      BindOutputs.push_back(
          BindOutput{CurInstr->fields[StoreIdx].ident, (int)Idx, 0, StoreIdx});
      MarkErase.push_back(Idx);
      NumCheckSafeToMove++;
      break;
    }

    case PatternNode::PN_ForkOther: {
      auto &AsForkOther = llvm::cast<ForkOtherNode>(*Node);
      assert(state == None || state == None_Ptr);
      ForkOtherOrigOp[&AsForkOther] = std::make_pair(Idx, OpIdx);
      process_impl(AsForkOther.Fork->Value.get(), Idx, state, OpIdx);
      break;
    }

    case PatternNode::PN_Constant: {
      auto &AsConstant = llvm::cast<ConstantNode>(*Node);
      PromoteToOperandMatcher();
      CheckConstantInt(AsConstant.Constant);
      break;
    }

    case PatternNode::PN_Load: {
      auto &AsLoad = llvm::cast<LoadNode>(*Node);
      PromoteToInsnMatcher();
      if (AsLoad.Size == (int)XLen)
        CheckOpcode("G_LOAD");
      else if (AsLoad.Sext)
        CheckOpcode("G_SEXTLOAD");
      else
        CheckOpcode("G_ZEXTLOAD");

      AddPredicate("AtomicOrderingMMOPredicateMatcher", {"\"NotAtomic\""});
      AddPredicate("MemorySizePredicateMatcher",
                   {"0", std::to_string(AsLoad.Size / 8)});

      process_impl(AsLoad.Addr.get(), Idx, None_Ptr, OperandIdx::fixed(1));
      break;
    }

    case PatternNode::PN_Cast: {
      auto &AsCast = llvm::cast<CastNode>(*Node);
      process_impl(AsCast.Value.get(), Idx, state, OpIdx);
      break;
    }

    case PatternNode::PN_Branch: {
      auto &AsBranch = llvm::cast<BranchNode>(*Node);
      CheckOpcode("G_BRCOND");

      auto *It = llvm::find_if(CurInstr->fields, [](auto &F) {
        return (int)F.type & (int)CDSLInstr::BRANCH_OFFS;
      });
      if (It == CurInstr->fields.end()) {
        Error = "Branch instruction but no branch offset immediate defined.";
        break;
      }

      auto IdxLast = Idx;
      state = None;
      OpIdx = OperandIdx::fixed(1);
      BindAndPromoteToOperandMatcher(It->ident);
      AddPredicate("MBBOperandMatcher", {});

      Idx = IdxLast;
      process_impl(AsBranch.Value.get(), Idx, None, OperandIdx::fixed(0));
      break;
    }

    default:
      Error = "unknown node";
      break;

      // case PatternNode::PN_NOp:
      // case PatternNode::PN_Ternop:
      // case PatternNode::PN_Shuffle:
      // case PatternNode::PN_Compare:
      // case PatternNode::PN_Unop:
      // case PatternNode::PN_Load:
      // case PatternNode::PN_Select:
      // case PatternNode::PN_Cast:
      // case PatternNode::PN_Store:
    }
  }

  void finalize() {
    if (NumCheckSafeToMove != 0)
      Output << "RM.addAction<CheckSafeToMoveInstAction>(" << NumCheckSafeToMove
             << ");\n";

    Output << "auto OutputInstID = RM.allocateOutputInsnID();\n";
    Output << "auto &DstI = Target.getInstruction(RK.getDef(\""
           << CurInstr->name
           << "_\"));\n"
              "auto &DstMIBuilder = RM.addAction<BuildMIAction>(OutputInstID, "
              "&DstI);\n";

    for (size_t i = 0; i < PatternArgs.size(); i++) {
      if (PatternArgs[i].Out)
        Output << "DstMIBuilder.addRenderer<CopyRenderer>(\""
               << CurInstr->fields[i].ident << (PatternArgs[i].In ? "_wb" : "")
               << "\");\n";
    }

    for (size_t i = 0; i < PatternArgs.size(); i++) {
      if (PatternArgs[i].In)
        Output << "DstMIBuilder.addRenderer<CopyRenderer>(\""
               << CurInstr->fields[i].ident << "\");\n";
    }

    size_t I = 0;
    for (auto &BindOutput : BindOutputs) {

      std::string BindName = BindOutput.Name.str();
      if (PatternArgs[BindOutput.OutInstOpIdx].In &&
          PatternArgs[BindOutput.OutInstOpIdx].Out)
        BindName += "_wb";

      Output << "auto &_O" << I << " = _" << BindOutput.SrcInstIdx
             << ".addOperand(" << BindOutput.SrcInstOpIdx << ", \"" << BindName
             << "\", 0);\n";
      Output << "_O" << I
             << ".addPredicate<RegisterBankOperandMatcher>(GPR);\n";
      // Output << "RM.addAction<ConstrainOperandToRegClassAction>(OutputInstID,
      // "
      //        << (CurInstr->fields.size() - 2 - BindOutput.OutInstOpIdx)
      //        << ", GPR);\n";
      I++;
    }

    Output << "RM.addAction<ConstrainOperandsToDefinitionAction>(0);\n";

    for (auto EraseIdx : MarkErase)
      Output << "RM.addAction<MarkEraseInstAction>(RM.getInsnVarID(_"
             << EraseIdx << "));\n";

    Output << "unsigned RootInsnID = "
              "RM.getInsnVarID(_0);\nRM.addAction<EraseInstAction>(RootInsnID);"
              "\nRules.push_back(std::move(RM));\npostProcessRule(Rules.back())"
              ";\n}\n";
  }
};

class ForkPermuter {
  RootNode *Root;
  std::vector<std::pair<ForkNode *, int>> Forks;
  std::vector<PatternNode *> RootStack;

  void findForks(PatternNode *Cur) {
    if (auto *AsFork = llvm::dyn_cast<ForkNode>(Cur)) {
      Forks.push_back(std::make_pair(AsFork, 0));
      findForks(AsFork->Value.get());
    }

    if (auto *AsRoot = llvm::dyn_cast<RootNode>(Cur))
      for (auto &Store : AsRoot->Stores)
        findForks(Store.second.get());

    for (auto &Op : Cur->getOperands())
      findForks(Op->get());
  }

  PatternNode *getRoot(PatternNode *Node) {
    assert(Node->Parent);
    if (llvm::isa<RootNode>(Node->Parent)) {
      return Node;
    }
    return getRoot(Node->Parent);
  }

  void pushRoots() {
    for (auto &Fork : Forks) {
      auto *Root = getRoot(Fork.first);
      if (llvm::find(RootStack, Root) == RootStack.end())
        RootStack.push_back(Root);
    }
  }

  PatternNode *findNewRoot(PatternNode *Node) {
    if (Node->Parent == Root)
      return Node;

    if (auto *AsForkOther = llvm::dyn_cast<ForkOtherNode>(Node))
      return findNewRoot(AsForkOther->Fork);

    return findNewRoot(Node->Parent);
  }

public:
  ForkPermuter(RootNode *Root) : Root(Root) { setup(); }

  void setup() {
    findForks(Root);
    pushRoots();
  }

  bool next() {

    if (!RootStack.empty()) {
      PatternNode *Cur = RootStack.back();
      RootStack.pop_back();

      auto It = llvm::find_if(
          Root->Stores, [&](auto &Pair) { return Pair.second.get() == Cur; });
      assert(It != Root->Stores.end());
      std::swap(*It, *Root->Stores.begin());

      return true;
    }

    for (size_t i = 0; i < Forks.size(); i++) {
      auto *Fork = Forks[i].first;
      auto *ForkOther =
          Fork->OtherUses[Forks[i].second % Fork->OtherUses.size()];

      Forks[i].second = (Forks[i].second + 1) % (Fork->OtherUses.size() + 1);

      {
        if (Fork->Parent == Root || ForkOther->Parent == Root)
          continue;

        auto *ForkOperand =
            *llvm::find_if(Fork->Parent->getOperands(),
                           [&](auto &Op) { return Op->get() == Fork; });
        auto *ForkOtherOperand =
            *llvm::find_if(ForkOther->Parent->getOperands(),
                           [&](auto &Op) { return Op->get() == ForkOther; });

        std::swap((*ForkOperand)->Parent, (*ForkOtherOperand)->Parent);
        std::swap(*ForkOperand, *ForkOtherOperand);
      }

      if (Forks[i].second != 0) {
        pushRoots();
        return next();
      }
    }
    return false;
  }
};

void GenGISelTable(RootNode *Root, std::ostream &Out) {

  ForkPermuter Permuter{Root};

  llvm::SmallDenseMap<int, std::pair<size_t, std::string>> BestCandidates;

  int I = 0;
  do {
    GISelTableBackend Backend{};
    llvm::outs() << "Permute: " << Root->patternString() << "\n";
    auto Score = Backend.process(Root);
    auto DstIdx = Root->Stores[0].first;
    auto Str = Backend.getOutput().str();

    if (auto Err = Backend.getError()) {
      llvm::errs() << "GISel pattern generation failed for permutation #" << I
                   << " of " << CurInstr->name << ": " << Err << "\n";
    } else {
      if (!BestCandidates.contains(DstIdx) ||
          Score < BestCandidates[DstIdx].first) {
        assert(Score != 0);
        BestCandidates[DstIdx] = std::make_pair(Score, Str);
      }
    }
    I++;
  } while (Permuter.next());

  for (auto &[Idx, Pair] : BestCandidates)
    Out << Pair.second;

  int Patterns = Root->Stores.size();
  int Success = BestCandidates.size();

  llvm::outs() << "Generated " << Success << " out of " << Patterns
               << " GISel pattern permutations for " << CurInstr->name << " ("
               << ((Success * 100) / Patterns) << "%)\n";
}

bool PatternGen::runOnMachineFunction(MachineFunction &MF) {

  // for convenience
  XLen = PatternGenArgs::Args.Is64Bit ? 64 : 32;
  RegT = PatternGenArgs::Args.Is64Bit ? "i64" : "i32";
  MayLoad = 0;
  MayStore = 0;
  IsBranch = 0;

  MF.dump();

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
  PatternArgs.append(CurInstr->fields.size() - 1, PatternArg());

  for (size_t I = 0; I < CurInstr->fields.size() - 1; I++)
    if (CurInstr->fields[I].type & CDSLInstr::BRANCH_OFFS) {
      PatternArgs[I].In = true;
      PatternArgs[I].ArgTypeStr = "simm13_lsb0";

      // very last argument is constant.
      if (I != CurInstr->fields.size() - 2) {
        llvm::errs() << "Pattern Generation failed for " << MF.getName() << ": "
                     << "Branch offset immediate is not last operand." << '\n';
        return true;
      }
    }

  PatternExtractor Extractor{};

  auto [Err, Node] = Extractor.generatePattern(MF);
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

  {
    if (PatternGenArgs::Args.GISelTableBackend)
      GenGISelTable(llvm::cast<RootNode>(Node.get()),
                    *PatternGenArgs::OutStreamGISelTable);
    else
      llvm::outs() << "Pattern for " << InstName << ": "
                   << Node->patternString() << '\n';
    ++PatternGenNumPatternsGenerated;

    LLT OutType = LLT();
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

        assert(!OutType.isValid() || PatternGenArgs::Args.GISelTableBackend);
        OutType = PatternArgs[I].Llt;
      }
    }

    InsString = InsString.substr(0, InsString.size() - 2);
    OutsString = OutsString.substr(0, OutsString.size() - 2);

    auto &OutStream = *PatternGenArgs::OutStream;

    OutStream << "let hasSideEffects = 0, mayLoad = " +
                     std::to_string((int)MayLoad) +
                     ", mayStore = " + std::to_string((int)MayStore) +
                     ", isBranch = " + std::to_string((int)IsBranch) +
                     ", isTerminator = " + std::to_string((int)IsBranch) +
                     ", isCodeGenOnly = 1";

    OutStream << ", Constraints = \"";
    {
      std::string Constr = "";
      for (size_t I = 0; I < CurInstr->fields.size() - 1; I++) {
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

    if (!PatternGenArgs::Args.GISelTableBackend) {
      std::string PatternStr = Node->patternString();
      std::string Code = "def : Pat<\n\t";

      if (OutType.isValid())
        Code += "(" + lltToString(OutType) + " " + PatternStr + "),\n\t(" +
                InstName + "_ ";
      else
        Code += PatternStr + ",\n\t(" + InstName + "_ ";

      Code += InsString;
      Code += ")>;";
      OutStream << "\n" << Code << "\n\n";
    }
  }

  // Delete all instructions to avoid match failures if patterns are not
  // included
  for (auto &MBB : MF)
    MBB.clear();

  return true;
}
