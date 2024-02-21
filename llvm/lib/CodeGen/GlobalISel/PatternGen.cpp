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
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Analysis/LazyBlockFrequencyInfo.h"
#include "llvm/Analysis/ProfileSummaryInfo.h"
#include "llvm/CodeGen/GlobalISel/GISelKnownBits.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/LegalizerInfo.h"
#include "llvm/CodeGen/GlobalISel/Utils.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/LowLevelType.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetOpcodes.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
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
#include <memory>
#include <sstream>
#include <utility>

#define DEBUG_TYPE "instruction-select"

using namespace llvm;

#ifdef LLVM_GISEL_COV_PREFIX
static cl::opt<std::string>
    CoveragePrefix("gisel-coverage-prefix", cl::init(LLVM_GISEL_COV_PREFIX),
                   cl::desc("Record GlobalISel rule coverage files of this "
                            "prefix if instrumentation was generated"));
#else
static const std::string CoveragePrefix;
#endif

std::ostream *PatternGenArgs::OutStream = nullptr;
std::string *PatternGenArgs::ExtName = nullptr;

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

static const std::unordered_map<unsigned, std::string> cmpStr = {
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
  std::string TypeStr;
  llvm::raw_string_ostream TypeStrS(TypeStr);
  Llt.print(TypeStrS);
  return TypeStr;
}

struct PatternNode {
  enum PatternNodeKind {
    PN_NOp,
    PN_Binop,
    PN_Shuffle,
    PN_Compare,
    PN_Unop,
    PN_Constant,
    PN_Register,
    PN_Select,
  };

private:
  const PatternNodeKind kind;

public:
  PatternNodeKind getKind() const { return kind; }
  LLT Type;
  PatternNode(PatternNodeKind Kind, LLT Type) : kind(Kind), Type(Type) {}

  virtual std::string patternString(int Indent = 0) = 0;
  virtual LLT getRegisterTy(int OperandId) const {
    if (OperandId == -1)
      return Type;
    return LLT(MVT::INVALID_SIMPLE_VALUE_TYPE);
  }
  virtual ~PatternNode() {}
};

struct NOpNode : public PatternNode {
  int Op;
  std::vector<std::unique_ptr<PatternNode>> Operands;
  NOpNode(LLT Type, int Op, std::vector<std::unique_ptr<PatternNode>> Operands)
      : PatternNode(PN_NOp, Type), Op(Op), Operands(std::move(Operands)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> BinopStr = {
        {TargetOpcode::G_BUILD_VECTOR, "build_vector"},
        {TargetOpcode::G_SELECT, "vselect"}};

    std::string S = "(" + std::string(BinopStr.at(Op)) + " ";
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
    return LLT(MVT::INVALID_SIMPLE_VALUE_TYPE);
  }
  static bool classof(const PatternNode *P) { return P->getKind() == PN_NOp; }
};

struct BinopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Left;
  std::unique_ptr<PatternNode> Right;

  BinopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Left,
            std::unique_ptr<PatternNode> Right)
      : PatternNode(PN_Binop, Type), Op(Op), Left(std::move(Left)),
        Right(std::move(Right)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> BinopStr = {
        {TargetOpcode::G_ADD, "add"},
        {TargetOpcode::G_SUB, "sub"},
        {TargetOpcode::G_MUL, "mul"},
        {TargetOpcode::G_SDIV, "div"},
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
        {TargetOpcode::G_EXTRACT_VECTOR_ELT, "vector_extract"}};

    std::string TypeStr = lltToString(Type);
    std::string OpString = "(" + std::string(BinopStr.at(Op)) + " " +
                           Left->patternString(Indent + 1) + ", " +
                           Right->patternString(Indent + 1) + ")";

    // Explicitly specifying types for all ops increases pattern compile time
    // significantly, so we only do for ops where deduction fails otherwise.
    bool PrintType = false;
    switch (Op) {
    case TargetOpcode::G_SHL:
    case TargetOpcode::G_LSHR:
    case TargetOpcode::G_ASHR:
      PrintType = true;
      break;
    default:
      break;
    }

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

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Binop; }
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

    return "(" + TypeStr + " (setcc " + Left->patternString(Indent + 1) + ", " +
           Right->patternString(Indent + 1) + ", " + cmpStr.at(Cond) + "))";
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
        Right(std::move(Right)), Tval(std::move(Tval)), Fval(std::move(Fval)) {}

  std::string patternString(int Indent = 0) override {
    std::string TypeStr = lltToString(Type);

    return "(" + TypeStr + " (riscv_selectcc " +
           Left->patternString(Indent + 1) + ", " +
           Right->patternString(Indent + 1) + ", " + cmpStr.at(Cond) + ", " +
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
    return LLT(MVT::INVALID_SIMPLE_VALUE_TYPE);
  }

  static bool classof(const PatternNode *p) {
    return p->getKind() == PN_Select;
  }
};

struct UnopNode : public PatternNode {
  int Op;
  std::unique_ptr<PatternNode> Operand;

  UnopNode(LLT Type, int Op, std::unique_ptr<PatternNode> Operand)
      : PatternNode(PN_Unop, Type), Op(Op), Operand(std::move(Operand)) {}

  std::string patternString(int Indent = 0) override {
    static const std::unordered_map<int, std::string> UnopStr = {
        {TargetOpcode::G_SEXT, "sext"},
        {TargetOpcode::G_ZEXT, "zext"},
        {TargetOpcode::G_VECREDUCE_ADD, "vecreduce_add"},
        {TargetOpcode::G_TRUNC, "trunc"},
        {TargetOpcode::G_BITCAST, "bitcast"},
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

  static bool classof(const PatternNode *p) { return p->getKind() == PN_Unop; }
};

struct ConstantNode : public PatternNode {
  uint32_t Constant;
  ConstantNode(LLT Type, uint32_t c)
      : PatternNode(PN_Constant, Type), Constant(c) {}

  std::string patternString(int Indent = 0) override {
    return "(i32 " + std::to_string(Constant) + ")";
  }

  static bool classof(const PatternNode *p) {
    return p->getKind() == PN_Constant;
  }
};

struct RegisterNode : public PatternNode {
  int RegId;
  int Offset; // in bytes
  int Size;   // 0 byte, 1 half, 2 word
  ISD::LoadExtType Ext;

  RegisterNode(LLT Type, int RegId, int Offset, int Size,
               ISD::LoadExtType Ext = ISD::LoadExtType::NON_EXTLOAD)
      : PatternNode(PN_Register, Type), RegId(RegId), Offset(Offset),
        Size(Size), Ext(Ext) {}

  std::string patternString(int Indent = 0) override {
    static const std::string RegNames[] = {"rd", "rs1", "rs2", "imm", "imm2"};

    // Immediate Operands
    if (RegId >= 3) {
      assert(Size == -1 && Offset == 0);
      return std::string("(i32 ") +
             (/*curInstr->SignedImm(regId - 3)*/ 0 ? "simm" : "uimm") +
             std::to_string(/*curInstr->GetImmLen(regId - 3)*/ 5) + ":$" +
             RegNames[RegId] + ")";
    }

    // Full-Size Register Operands
    if (Size == 2) {
      if (Type.isScalar() && Type.getSizeInBits() == 32)
        return "GPR:$" + RegNames[RegId];
      if (Type.isFixedVector() && Type.getSizeInBits() == 32 &&
          Type.getElementType().isScalar() &&
          Type.getElementType().getSizeInBits() == 8)
        return "PulpV4:$" + RegNames[RegId];
      if (Type.isFixedVector() && Type.getSizeInBits() == 32 &&
          Type.getElementType().isScalar() &&
          Type.getElementType().getSizeInBits() == 16)
        return "PulpV2:$" + RegNames[RegId];

      abort();
    }

    // Sub-Register Operands
    if (Size == 1 || Size == 0) {
      std::string Str;
      if (Type.isScalar() && Type.getSizeInBits() == 32) {
        assert(Offset == 0);
        Str = "GPR:$" + RegNames[RegId];
      } else
        Str = std::string("(i32 (vector_extract PulpV") + (Size ? "2" : "4") +
              ":$" + RegNames[RegId] + ", " +
              std::to_string(Size ? (Offset / 2) : (Offset)) + "))";

      std::string Mask = Size ? "65535" : "255";
      std::string Shamt = Size ? "16" : "24";
      switch (Ext) {
      case ISD::LoadExtType::EXTLOAD:
        return Str;
      case ISD::LoadExtType::ZEXTLOAD:
        return "(and " + Str + ", (i32 " + Mask + "))";
      case ISD::LoadExtType::SEXTLOAD:
        return "(sra (shl " + Str + ", (i32 " + Shamt + ")), (i32 " + Shamt +
               "))";
      default:
        break;
      }
    }
    abort();
  }

  LLT getRegisterTy(int OperandId) const override {
    if (OperandId == -1)
      return Type;
    if (OperandId == RegId)
      return Type;
    return LLT();
  }

  static bool classof(const PatternNode *p) {
    return p->getKind() == PN_Register;
  }
};

static std::pair<PatternError, std::unique_ptr<PatternNode>>
traverse(MachineRegisterInfo &MRI, MachineInstr &Cur);

static std::pair<PatternError, std::unique_ptr<PatternNode>>
traverseOperand(MachineRegisterInfo &MRI, MachineInstr &Cur, int i) {
  auto *Op = MRI.getOneDef(Cur.getOperand(1).getReg());
  if (!Op)
    return std::make_pair(FORMAT, nullptr);
  auto [Err, Node] = traverse(MRI, *Op->getParent());
  if (Err)
    return std::make_pair(Err, nullptr);

  return std::make_pair(SUCCESS, std::move(Node));
}

static std::tuple<PatternError, std::unique_ptr<PatternNode>,
                  std::unique_ptr<PatternNode>>
traverseBinopOperands(MachineRegisterInfo &MRI, MachineInstr &Cur,
                      int start = 1) {
  auto *LHS = MRI.getOneDef(Cur.getOperand(start).getReg());
  if (!LHS)
    return std::make_tuple(PatternError(FORMAT, &Cur), nullptr, nullptr);
  auto *RHS = MRI.getOneDef(Cur.getOperand(start + 1).getReg());
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

static std::pair<PatternError, std::unique_ptr<PatternNode>>

traverse(MachineRegisterInfo &MRI, MachineInstr &Cur) {

  switch (Cur.getOpcode()) {
  case TargetOpcode::G_ADD:
  case TargetOpcode::G_SUB:
  case TargetOpcode::G_MUL:
  case TargetOpcode::G_SDIV:
  case TargetOpcode::G_UDIV:
  case TargetOpcode::G_SREM:
  case TargetOpcode::G_UREM:
  case TargetOpcode::G_AND:
  case TargetOpcode::G_OR:
  case TargetOpcode::G_XOR:
  case TargetOpcode::G_SHL:
  case TargetOpcode::G_LSHR:
  case TargetOpcode::G_ASHR: {

    auto [Err, NodeL, NodeR] = traverseBinopOperands(MRI, Cur);
    if (Err)
      return std::make_pair(Err, nullptr);

    auto Node = std::make_unique<BinopNode>(
        MRI.getType(Cur.getOperand(0).getReg()), Cur.getOpcode(),
        std::move(NodeL), std::move(NodeR));

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_BITCAST: {
    // Bitcasts are normally transparent, but they affect the register
    // type for G_LOAD (handled via fallthrough)
    auto *Operand = MRI.getOneDef(Cur.getOperand(1).getReg());
    if (!Operand)
      return std::make_pair(PatternError(FORMAT_LOAD, &Cur), nullptr);

    auto [Err, Node] = traverse(MRI, *Operand->getParent());
    if (Err)
      return std::make_pair(Err, nullptr);

    // if the bitcasted value is a register access, we need to patch the
    // register access type
    if (auto *AsRegNode = llvm::dyn_cast<RegisterNode>(Node.get()))
      AsRegNode->Type = MRI.getType(Cur.getOperand(0).getReg());

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_LOAD: {
    auto *Addr = MRI.getOneDef(Cur.getOperand(1).getReg());
    if (!Addr)
      return std::make_pair(PatternError(FORMAT_LOAD, &Cur), nullptr);
    auto *AddrI = Addr->getParent();
    if (AddrI->getOpcode() != TargetOpcode::COPY)
      return std::make_pair(PatternError(FORMAT_LOAD, AddrI), nullptr);

    auto AddrLI = AddrI->getOperand(1).getReg();
    if (!MRI.isLiveIn(AddrLI) || !AddrLI.isPhysical())
      return std::make_pair(PatternError(FORMAT_LOAD, AddrI), nullptr);

    auto Node =
        std::make_unique<RegisterNode>(MRI.getType(Cur.getOperand(0).getReg()),
                                       AddrLI.asMCReg().id() - 41 - 10, 0, 2);

    return std::make_pair(SUCCESS, std::move(Node));
  }
  case TargetOpcode::G_CONSTANT: {
    auto *Imm = Cur.getOperand(1).getCImm();
    return std::make_pair(SUCCESS, std::make_unique<ConstantNode>(
                                       MRI.getType(Cur.getOperand(0).getReg()),
                                       Imm->getLimitedValue()));
  }
  case TargetOpcode::G_ICMP: {
    auto Pred = Cur.getOperand(1);
    auto [Err, NodeL, NodeR] = traverseBinopOperands(MRI, Cur, 2);
    if (Err)
      return std::make_pair(Err, nullptr);

    return std::make_pair(SUCCESS, std::make_unique<CompareNode>(
                                       MRI.getType(Cur.getOperand(0).getReg()),
                                       (CmpInst::Predicate)Pred.getPredicate(),
                                       std::move(NodeL), std::move(NodeR)));
  }
  case TargetOpcode::COPY: {
    // Immediate Operands
    auto Reg = Cur.getOperand(1).getReg();
    if (!Reg.isPhysical())
      return std::make_pair(FORMAT_IMM, nullptr);

    uint Id = Reg.asMCReg().id() - 41 - 10;
    if (!(Id >= 3 && Id <= 4))
      return std::make_pair(FORMAT_IMM, nullptr);
    
    return std::make_pair(
        SUCCESS, std::make_unique<RegisterNode>(
                     MRI.getType(Cur.getOperand(0).getReg()), Id, 0, -1));
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

  auto *Addr = MRI.getOneDef(Store.getOperand(1).getReg());
  if (Addr == nullptr)
    return std::make_pair(FORMAT_STORE, nullptr);
  // todo: check that address is first argument (x10)

  auto *Root = MRI.getOneDef(Store.getOperand(0).getReg());
  if (Root == nullptr)
    return std::make_pair(FORMAT_STORE, nullptr);

  return traverse(MRI, *Root->getParent());
}

bool PatternGen::runOnMachineFunction(MachineFunction &MF) {

  auto [Err, Node] = generatePattern(MF);
  if (Err) {
    llvm::outs() << "Pattern Generation failed for " << MF.getName() << ": "
                 << Errors[Err.Type] << '\n';
    if (Err.Inst) {
      llvm::outs() << "Match failure occurred here:\n";
      llvm::outs() << *Err.Inst << "\n";
    }
    return true;
  }

  std::string InstName = MF.getName().str().substr(4);
  std::string InstNameO = InstName;

  llvm::outs() << "Pattern for " << InstName << ": " << Node->patternString()
               << '\n';

  // types for: rd, rd (as src), rs1, rs2, imm, imm2
  std::array<std::string, 6> TypeStrings = {"", "", "", "", "", ""};
  for (int I = 0; I < 4; I++) {
    auto Type = Node->getRegisterTy(I - 1);
    if (Type.isValid()) {
      if (Type.isFixedVector() && Type.getElementType().isScalar() &&
          Type.getSizeInBits() == 32) {
        if (Type.getElementType().getSizeInBits() == 8) {
          TypeStrings[I] = "PulpV4";
          InstName += "_V4";
        } else if (Type.getElementType().getSizeInBits() == 16) {
          TypeStrings[I] = "PulpV2";
          InstName += "_V2";
        } else
          abort();
      } else {
        TypeStrings[I] = "GPR";
        InstName += "_S";
      }
    }
  }

  std::string OutsString =
      TypeStrings[0] + (TypeStrings[1].empty() ? ":$rd" : ":$rd_wb");
  std::string InsString;
  static const auto OpNames =
      std::array<std::string, 5>{"rd", "rs1", "rs2", "imm", "imm2"};
  for (int I = 1; I < 6; I++)
    if (!TypeStrings[I].empty()) {
      InsString += TypeStrings[I] + ":$" + (OpNames[I - 1]) + ", ";
    }
  InsString = InsString.substr(0, InsString.size() - 2);

  auto &OutStream = *PatternGenArgs::OutStream;

  OutStream << "let Predicates = [HasExt"
               "Xcvsimd"
               "], hasSideEffects = 0, mayLoad = 0, mayStore = 0, "
               "isCodeGenOnly = 1";
  if (!TypeStrings[1].empty())
    OutStream << ", Constraints = \"$rd = $rd_wb\"";
  OutStream << " in ";
  OutStream << "def " << InstName << " : RVInst_" << InstNameO << "<(outs "
            << OutsString << "), (ins " << InsString << ")>;\n";

  std::string PatternStr = Node->patternString();
  std::string Code = "def : Pat<\n\t" + PatternStr + ",\n\t(" + InstName + " ";

  Code += InsString;
  Code += ")>;";
  OutStream << "\n" << Code << "\n\n";

  // Delete all instructions to avoid match failures if patterns are not
  // included
  for (auto &MBB : MF)
    MBB.clear();

  return true;
}
