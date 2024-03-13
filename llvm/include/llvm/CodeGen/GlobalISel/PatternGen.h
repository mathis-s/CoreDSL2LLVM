//== llvm/CodeGen/GlobalISel/PatternGen.h -----------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file This file describes the interface of the MachineFunctionPass
/// responsible for selecting (possibly generic) machine instructions to
/// target-specific instructions.
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_GLOBALISEL_PATTERNGEN_H
#define LLVM_CODEGEN_GLOBALISEL_PATTERNGEN_H

#include "../../../tools/pattern-gen/PatternGen.hpp"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CodeGen.h"

struct CDSLInstr;

namespace llvm {

namespace PatternGenArgs {
extern std::ostream *OutStream;
extern std::vector<CDSLInstr> const *Instrs;
extern PGArgsStruct Args;
} // namespace PatternGenArgs

class BlockFrequencyInfo;
class ProfileSummaryInfo;

class PatternGen : public MachineFunctionPass {
public:
  static char ID;
  StringRef getPassName() const override { return "PatternGen"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  MachineFunctionProperties getRequiredProperties() const override {
    return MachineFunctionProperties()
        .set(MachineFunctionProperties::Property::IsSSA)
        .set(MachineFunctionProperties::Property::Legalized)
        .set(MachineFunctionProperties::Property::RegBankSelected);
  }

  PatternGen(CodeGenOptLevel OL);
  PatternGen();

  bool runOnMachineFunction(MachineFunction &MF) override;

protected:
  BlockFrequencyInfo *BFI = nullptr;
  ProfileSummaryInfo *PSI = nullptr;

  CodeGenOptLevel OptLevel = CodeGenOptLevel::None;
};
} // End namespace llvm.

#endif
