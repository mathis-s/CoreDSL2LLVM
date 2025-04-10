for (int R_0 = 0; R_0 < 2; R_0++)
for (int R_1 = 0; R_1 < 2; R_1++)
{
RuleMatcher RM{Locs};
RuleMatcherScores[RM.getRuleID()] = 16;
RM.addRequiredFeature(RK.getDef("HasVendorXCValu"));
RM.addRequiredFeature(RK.getDef("IsRV32"));
InstructionMatcher &_0 = RM.addInstructionMatcher("");
_0.addPredicate<InstructionOpcodeMatcher>(&Target.getInstruction(RK.getDef("G_BRCOND")));
auto &_1 = _0.addOperand(1, "imm12", 0);
_1.addPredicate<MBBOperandMatcher>();
auto &_2 = _0.addOperand(0, "", 0);
auto &_3 = (**(_2.addPredicate<InstructionOperandMatcher>(RM, ""))).getInsnMatcher();
_3.addPredicate<InstructionOpcodeMatcher>(&Target.getInstruction(RK.getDef("G_ICMP")));
_3.addPredicate<CmpPredicateOperandMatcher>(1, "ICMP_NE");
auto &_4 = _3.addOperand((2+(R_0+0)%2), "", 0);
_4.addPredicate<ConstantIntOperandMatcher>(0);
auto &_5 = _3.addOperand((2+(R_0+1)%2), "", 0);
auto &_6 = (**(_5.addPredicate<InstructionOperandMatcher>(RM, ""))).getInsnMatcher();
_6.addPredicate<InstructionOpcodeMatcher>(&Target.getInstruction(RK.getDef("G_AND")));
auto &_7 = _6.addOperand((1+(R_1+0)%2), "rs2", 0);
_7.addPredicate<LLTOperandMatcher>(LLT::scalar(32));
_7.addPredicate<RegisterBankOperandMatcher>(GPR);
auto &_8 = _6.addOperand((1+(R_1+1)%2), "rs1", 0);
_8.addPredicate<LLTOperandMatcher>(LLT::scalar(32));
_8.addPredicate<RegisterBankOperandMatcher>(GPR);
auto OutputInstID = RM.allocateOutputInsnID();
auto &DstI = Target.getInstruction(RK.getDef("BMASK_"));
auto &DstMIBuilder = RM.addAction<BuildMIAction>(OutputInstID, &DstI);
DstMIBuilder.addRenderer<CopyRenderer>("rs1");
DstMIBuilder.addRenderer<CopyRenderer>("rs2");
DstMIBuilder.addRenderer<CopyRenderer>("imm12");
unsigned RootInsnID = RM.getInsnVarID(_0);
RM.addAction<EraseInstAction>(RootInsnID);
Rules.push_back(std::move(RM));
postProcessRule(Rules.back());
}
