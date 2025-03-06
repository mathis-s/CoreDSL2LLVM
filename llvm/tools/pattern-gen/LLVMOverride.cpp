/*
This file contains some code duplicated or slightly adjusted from LLVM,
mostly overriding specific virtual functions to inject our own code.

The alternative to using a file like this is modifying LLVM source
more aggressively directly.
*/
#include "../lib/Target/RISCV/RISCVISelDAGToDAG.h"
#include "../lib/Target/RISCV/RISCVTargetMachine.h"
#include "PatternGen.hpp"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/CodeGen/CommandFlags.h"
#include "llvm/CodeGen/FunctionLoweringInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/PatternGen.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/ISDOpcodes.h"
#include "llvm/CodeGen/LinkAllAsmWriterComponents.h"
#include "llvm/CodeGen/LinkAllCodegenComponents.h"
#include "llvm/CodeGen/MIRParser/MIRParser.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/MachineScheduler.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/SelectionDAG.h"
#include "llvm/CodeGen/SelectionDAGISel.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DebugLoc.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/DiagnosticPrinter.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/Timer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/Scalar.h"
#include <cctype>
#include <ostream>
#define DEBUG_TYPE "isel"

using namespace llvm;
static codegen::RegisterCodeGenFlags CGF;

static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

static bool EnableRedundantCopyElimination = true;
static auto EnableGlobalMerge = cl::BOU_UNSET;
static bool EnableMachineCombiner = true;
static bool EnableRISCVCopyPropagation = true;
static bool EnableRISCVDeadRegisterElimination = true;
static bool EnableSinkFold = true;
static bool EnableLoopDataPrefetch = true;
static bool EnableMISchedLoadClustering = false;
static llvm::once_flag InitializeDefaultRVVRegisterAllocatorFlag;
static auto RVVRegAlloc = useDefaultRegisterAllocator;
static bool EnableVSETVLIAfterRVVRegAlloc = true;

namespace {

class RVVRegisterRegAlloc : public RegisterRegAllocBase<RVVRegisterRegAlloc> {
public:
  RVVRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

static bool onlyAllocateRVVReg(const TargetRegisterInfo &TRI,
                               const MachineRegisterInfo &MRI,
                               const Register Reg) {
  const TargetRegisterClass *RC = MRI.getRegClass(Reg);
  return RISCVRegisterInfo::isRVVRegClass(RC);
}

static void initializeDefaultRVVRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = RVVRegAlloc;
    RVVRegisterRegAlloc::setDefault(RVVRegAlloc);
  }
}

static FunctionPass *createGreedyRVVRegisterAllocator() {
  return createGreedyRegisterAllocator(onlyAllocateRVVReg);
}

static FunctionPass *createFastRVVRegisterAllocator() {
  return createFastRegisterAllocator(onlyAllocateRVVReg, false);
}

class RISCVPassConfig : public TargetPassConfig {
public:
  RISCVPassConfig(RISCVTargetMachine &TM, PassManagerBase &PM)
      : TargetPassConfig(TM, PM) {
    if (TM.getOptLevel() != CodeGenOptLevel::None)
      substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
    setEnableSinkAndFold(EnableSinkFold);
  }

  RISCVTargetMachine &getRISCVTargetMachine() const {
    return getTM<RISCVTargetMachine>();
  }

  ScheduleDAGInstrs *
  createMachineScheduler(MachineSchedContext *C) const override {
    ScheduleDAGMILive *DAG = nullptr;
    if (EnableMISchedLoadClustering) {
      DAG = createGenericSchedLive(C);
      DAG->addMutation(createLoadClusterDAGMutation(
          DAG->TII, DAG->TRI, /*ReorderWhileClustering=*/true));
    }
    return DAG;
  }

  void addIRPasses() override;
  bool addPreISel() override;
  void addCodeGenPrepare() override;
  bool addInstSelector() override;
  bool addIRTranslator() override;
  void addPreLegalizeMachineIR() override;
  bool addLegalizeMachineIR() override;
  void addPreRegBankSelect() override;
  bool addRegBankSelect() override;
  bool addGlobalInstructionSelect() override;
  void addPreEmitPass() override;
  void addPreEmitPass2() override;
  void addPreSched2() override;
  void addMachineSSAOptimization() override;
  FunctionPass *createRVVRegAllocPass(bool Optimized);
  bool addRegAssignAndRewriteFast() override;
  bool addRegAssignAndRewriteOptimized() override;
  void addPreRegAlloc() override;
  void addPostRegAlloc() override;
  void addFastRegAlloc() override;
};
} // namespace

FunctionPass *RISCVPassConfig::createRVVRegAllocPass(bool Optimized) {
  // Initialize the global default.
  llvm::call_once(InitializeDefaultRVVRegisterAllocatorFlag,
                  initializeDefaultRVVRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = RVVRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  if (Optimized)
    return createGreedyRVVRegisterAllocator();

  return createFastRVVRegisterAllocator();
}

bool RISCVPassConfig::addRegAssignAndRewriteFast() {
  addPass(createRVVRegAllocPass(false));
  if (EnableVSETVLIAfterRVVRegAlloc)
    addPass(createRISCVInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableRISCVDeadRegisterElimination)
    addPass(createRISCVDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteFast();
}

bool RISCVPassConfig::addRegAssignAndRewriteOptimized() {
  addPass(createRVVRegAllocPass(true));
  addPass(createVirtRegRewriter(false));
  if (EnableVSETVLIAfterRVVRegAlloc)
    addPass(createRISCVInsertVSETVLIPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableRISCVDeadRegisterElimination)
    addPass(createRISCVDeadRegisterDefinitionsPass());
  return TargetPassConfig::addRegAssignAndRewriteOptimized();
}

void RISCVPassConfig::addIRPasses() {
  addPass(createAtomicExpandLegacyPass());

  if (getOptLevel() != CodeGenOptLevel::None) {
    if (EnableLoopDataPrefetch)
      addPass(createLoopDataPrefetchPass());

    addPass(createRISCVGatherScatterLoweringPass());
    addPass(createInterleavedAccessPass());
    addPass(createRISCVCodeGenPreparePass());
  }

  TargetPassConfig::addIRPasses();
}

bool RISCVPassConfig::addPreISel() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    // Add a barrier before instruction selection so that we will not get
    // deleted block address after enabling default outlining. See D99707 for
    // more details.
    addPass(createBarrierNoopPass());
  }

  if (EnableGlobalMerge == cl::BOU_TRUE) {
    addPass(createGlobalMergePass(TM, /* MaxOffset */ 2047,
                                  /* OnlyOptimizeForSize */ false,
                                  /* MergeExternalByDefault */ true));
  }

  return false;
}

void RISCVPassConfig::addCodeGenPrepare() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createTypePromotionLegacyPass());
  TargetPassConfig::addCodeGenPrepare();
}

bool RISCVPassConfig::addInstSelector() {
  addPass(createRISCVISelDag(getRISCVTargetMachine(), getOptLevel()));

  return false;
}

bool RISCVPassConfig::addIRTranslator() {
  addPass(new IRTranslator(getOptLevel()));
  return false;
}

void RISCVPassConfig::addPreLegalizeMachineIR() {
  if (getOptLevel() == CodeGenOptLevel::None) {
    addPass(createRISCVO0PreLegalizerCombiner());
  } else {
    addPass(createRISCVPreLegalizerCombiner());
  }
}

bool RISCVPassConfig::addLegalizeMachineIR() {
  addPass(new Legalizer());
  return false;
}

void RISCVPassConfig::addPreRegBankSelect() {
  if (getOptLevel() != CodeGenOptLevel::None)
    addPass(createRISCVPostLegalizerCombiner());
}

bool RISCVPassConfig::addRegBankSelect() {
  addPass(new RegBankSelect());
  return false;
}

bool RISCVPassConfig::addGlobalInstructionSelect() {
  addPass(new InstructionSelect(getOptLevel()));
  return false;
}

void RISCVPassConfig::addPreSched2() {
  addPass(createRISCVPostRAExpandPseudoPass());

  // Emit KCFI checks for indirect calls.
  addPass(createKCFIPass());
}

void RISCVPassConfig::addPreEmitPass() {
  // TODO: It would potentially be better to schedule copy propagation after
  // expanding pseudos (in addPreEmitPass2). However, performing copy
  // propagation after the machine outliner (which runs after addPreEmitPass)
  // currently leads to incorrect code-gen, where copies to registers within
  // outlined functions are removed erroneously.
  if (TM->getOptLevel() >= CodeGenOptLevel::Default &&
      EnableRISCVCopyPropagation)
    addPass(createMachineCopyPropagationPass(true));
  addPass(&BranchRelaxationPassID);
  addPass(createRISCVMakeCompressibleOptPass());
}

void RISCVPassConfig::addPreEmitPass2() {
  if (TM->getOptLevel() != CodeGenOptLevel::None) {
    addPass(createRISCVMoveMergePass());
    // Schedule PushPop Optimization before expansion of Pseudo instruction,
    // ensuring return instruction is detected correctly.
    addPass(createRISCVPushPopOptimizationPass());
  }
  addPass(createRISCVExpandPseudoPass());

  // Schedule the expansion of AMOs at the last possible moment, avoiding the
  // possibility for other passes to break the requirements for forward
  // progress in the LR/SC block.
  addPass(createRISCVExpandAtomicPseudoPass());

  // KCFI indirect call checks are lowered to a bundle.
  addPass(createUnpackMachineBundles([&](const MachineFunction &MF) {
    return MF.getFunction().getParent()->getModuleFlag("kcfi");
  }));
}

void RISCVPassConfig::addMachineSSAOptimization() {
  addPass(createRISCVVectorPeepholePass());

  TargetPassConfig::addMachineSSAOptimization();

  if (EnableMachineCombiner)
    addPass(&MachineCombinerID);

  if (TM->getTargetTriple().isRISCV64()) {
    addPass(createRISCVOptWInstrsPass());
  }
}

void RISCVPassConfig::addPreRegAlloc() {
  addPass(createRISCVPreRAExpandPseudoPass());
  if (TM->getOptLevel() != CodeGenOptLevel::None)
    addPass(createRISCVMergeBaseOffsetOptPass());

  addPass(createRISCVInsertReadWriteCSRPass());
  addPass(createRISCVInsertWriteVXRMPass());

  // Run RISCVInsertVSETVLI after PHI elimination. On O1 and above do it after
  // register coalescing so needVSETVLIPHI doesn't need to look through COPYs.
  if (!EnableVSETVLIAfterRVVRegAlloc) {
    if (TM->getOptLevel() == CodeGenOptLevel::None)
      insertPass(&PHIEliminationID, &RISCVInsertVSETVLIID);
    else
      insertPass(&RegisterCoalescerID, &RISCVInsertVSETVLIID);
  }
}

void RISCVPassConfig::addFastRegAlloc() {
  addPass(&InitUndefID);
  TargetPassConfig::addFastRegAlloc();
}

void RISCVPassConfig::addPostRegAlloc() {
  if (TM->getOptLevel() != CodeGenOptLevel::None &&
      EnableRedundantCopyElimination)
    addPass(createRISCVRedundantCopyEliminationPass());
}

class RISCVPatternPassConfig : public RISCVPassConfig {
public:
  RISCVPatternPassConfig(RISCVTargetMachine &TM, PassManagerBase &PM)
      : RISCVPassConfig(TM, PM) {}

  bool addGlobalInstructionSelect() override {
    addPass(new PatternGen());
    return this->RISCVPassConfig::addGlobalInstructionSelect();
  }
};

class RISCVPatternsTargetMachine : public RISCVTargetMachine {
public:
  RISCVPatternsTargetMachine(const Target &T, const Triple &TT, StringRef CPU,
                             StringRef FS, const TargetOptions &Options,
                             std::optional<Reloc::Model> RM,
                             std::optional<CodeModel::Model> CM,
                             CodeGenOptLevel OL, bool JIT)
      : RISCVTargetMachine(T, TT, CPU, FS, Options, RM, CM, OL, JIT) {}

  TargetPassConfig *createPassConfig(PassManagerBase &PM) override {
    return new RISCVPatternPassConfig(*this, PM);
  }
};

void optimizeModule(llvm::TargetMachine *Machine, llvm::Module *Mod,
                    llvm::CodeGenOptLevel OptLevel) {
  Mod->setTargetTriple(Machine->getTargetTriple().str());
  Mod->setDataLayout(Machine->createDataLayout());

  // Create the analysis managers.
  LoopAnalysisManager LAM;
  FunctionAnalysisManager FAM;
  CGSCCAnalysisManager CGAM;
  ModuleAnalysisManager MAM;
  PipelineTuningOptions PTO;
  PTO.SLPVectorization = OptLevel > llvm::CodeGenOptLevel::None;

  // Create the new pass manager builder.
  // Take a look at the PassBuilder constructor parameters for more
  // customization, e.g. specifying a TargetMachine or various debugging
  // options.
  PassBuilder PB(Machine, PTO);

  // Register all the basic analyses with the managers.
  PB.registerModuleAnalyses(MAM);
  PB.registerCGSCCAnalyses(CGAM);
  PB.registerFunctionAnalyses(FAM);
  PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  // Create the pass manager.
  // This one corresponds to a typical -O2 optimization pipeline.
  ModulePassManager MPM =
      PB.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);

  // Optimize the IR!
  MPM.run(*Mod, MAM);
}

static void setOptions() {
  const char *Args[] = {"", "--global-isel", "--global-isel-abort=1"};
  cl::ParseCommandLineOptions(sizeof(Args) / sizeof(Args[0]), Args);
}

std::unique_ptr<TargetMachine> getTargetMachine(bool Is64Bit,
                                                std::string Mattr) {
  setOptions();
  SMDiagnostic Err;
  Triple TheTriple((Is64Bit ? "riscv64" : "riscv32"), "unknown", "linux",
                   "gnu");

  TargetOptions Options = codegen::InitTargetOptionsFromCodeGenFlags(TheTriple);
  std::string CPUStr = codegen::getCPUStr(),
              FeaturesStr = codegen::getFeaturesStr() + Mattr;

  auto MAttrs = codegen::getMAttrs();

  CodeGenOptLevel OLvl = CodeGenOptLevel::Aggressive;
  Options.EnableGlobalISel = true;

  std::optional<Reloc::Model> RM = codegen::getExplicitRelocModel();
  std::optional<CodeModel::Model> CM = codegen::getExplicitCodeModel();

  const Target *TheTarget = nullptr;
  std::unique_ptr<TargetMachine> Target;

  // Get the target specific parser.
  std::string Error;
  TheTarget =
      TargetRegistry::lookupTarget(codegen::getMArch(), TheTriple, Error);
  assert(TheTarget);

  Target = std::make_unique<RISCVPatternsTargetMachine>(
      *TheTarget, TheTriple, CPUStr, FeaturesStr, Options, RM, CM, OLvl, false);
  // Target = std::unique_ptr<TargetMachine>(TheTarget->createTargetMachine(
  //     TheTriple.getTriple(), CPUStr, FeaturesStr, Options, RM, CM, OLvl));
  assert(Target && "Could not allocate target machine!");

  return Target;
}

int runOptPipeline(llvm::Module *M, bool Is64Bit, std::string Mattr,
                   llvm::CodeGenOptLevel OptLevel, std::ostream &IrOut) {

  InitializeAllTargets();
  InitializeAllTargetMCs();
  InitializeAllAsmPrinters();
  InitializeAllAsmParsers();

  PassRegistry *Registry = PassRegistry::getPassRegistry();
  initializeCore(*Registry);
  initializeCodeGen(*Registry);
  initializeLoopStrengthReducePass(*Registry);
  initializeLowerIntrinsicsPass(*Registry);
  initializePostInlineEntryExitInstrumenterPass(*Registry);
  initializeUnreachableBlockElimLegacyPassPass(*Registry);
  initializeConstantHoistingLegacyPassPass(*Registry);
  initializeScalarOpts(*Registry);
  initializeVectorization(*Registry);
  initializeScalarizeMaskedMemIntrinLegacyPassPass(*Registry);
  initializeExpandReductionsPass(*Registry);
  initializeHardwareLoopsLegacyPass(*Registry);
  initializeTransformUtils(*Registry);
  initializeReplaceWithVeclibLegacyPass(*Registry);

  // Initialize debugging passes.
  initializeScavengerTestPass(*Registry);

  // Load the module to be compiled...
  // SMDiagnostic Err;
  auto Target = getTargetMachine(Is64Bit, Mattr);

  M->setDataLayout(Target->createDataLayout().getStringRepresentation());

  optimizeModule(Target.get(), M, OptLevel);
  {
    std::string ModuleStr;
    {
      llvm::raw_string_ostream StrStream(ModuleStr);
      StrStream << *M;
    }
    IrOut << ModuleStr;
  }
  return 0;
}

int runPatternGenPipeline(llvm::Module *M, bool Is64Bit, std::string Mattr) {

  auto Target = getTargetMachine(Is64Bit, Mattr);

  if (codegen::getFloatABIForCalls() != FloatABI::Default)
    Target->Options.FloatABIType = codegen::getFloatABIForCalls();

  std::unique_ptr<MIRParser> MIR;

  // Figure out where we are going to send the output.
  // std::unique_ptr<ToolOutputFile> Out = GetOutputStream(TheTarget->getName(),
  // TheTriple.getOS(), "pattern-gen"); if (!Out) return 1;

  // Ensure the filename is passed down to CodeViewDebug.
  // Target->Options.ObjectFilenameForDebug = Out->outputFilename();

  // Add an appropriate TargetLibraryInfo pass for the module's triple.
  TargetLibraryInfoImpl TLII(Triple(M->getTargetTriple()));

  // Verify module immediately to catch problems before doInitialization() is
  // called on any passes.
  assert(!verifyModule(*M, &errs()));

  // Override function attributes based on CPUStr, FeaturesStr, and command line
  // flags.
  codegen::setFunctionAttributes(Target->getTargetCPU(),
                                 Target->getTargetFeatureString(), *M);

  // if (EnableNewPassManager || !PassPipeline.empty()) {
  //   return compileModuleWithNewPM(argv[0], std::move(M), std::move(MIR),
  //                                 std::move(Target), std::move(Out),
  //                                 std::move(DwoOut), Context, TLII, NoVerify,
  //                                 PassPipeline, codegen::getFileType());
  // }

  // Build up all of the passes that we want to do to the module.
  legacy::PassManager PM;
  PM.add(new TargetLibraryInfoWrapperPass(TLII));

  {
    SmallVector<char> Out;
    raw_svector_ostream SVOS{Out};
    raw_pwrite_stream *OS = &SVOS;

    // LLVMTargetMachine &LLVMTM = static_cast<LLVMTargetMachine &>(*Target);
    MachineModuleInfoWrapperPass *MMIWP =
        new MachineModuleInfoWrapperPass(Target.get());
        // new MachineModuleInfoWrapperPass(&LLVMTM);

    // Construct a custom pass pipeline that starts after instruction
    // selection.
    if (Target->addPassesToEmitFile(PM, *OS, nullptr, codegen::getFileType(),
                                    false, MMIWP)) {
      assert(0 && "target does not support generation of this file type");
    }

    // const_cast<TargetLoweringObjectFile *>(LLVMTM.getObjFileLowering())
    const_cast<TargetLoweringObjectFile *>(Target->getObjFileLowering())
        ->Initialize(MMIWP->getMMI().getContext(), *Target);
    if (MIR) {
      assert(MMIWP && "Forgot to create MMIWP?");
      if (MIR->parseMachineFunctions(*M, MMIWP->getMMI()))
        return 1;
    }

    // Before executing passes, print the final values of the LLVM options.
    cl::PrintOptionValues();

    PM.run(*M);
  }

  return 0;
}
