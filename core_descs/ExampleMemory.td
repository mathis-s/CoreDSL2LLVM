let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LDR_ : RVInst_LDR<(outs GPR:$rd), (ins uimm5:$imm, GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (load (ptradd (iPTR GPR:$rs1), (i64 (shl (i64 GPR:$rs2), (i64 (i64 uimm5:$imm)))))))),
	(LDR_ uimm5:$imm, GPR:$rs2, GPR:$rs1)>;
}
