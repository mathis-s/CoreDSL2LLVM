let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LDR_ : RVInst_LDR<(outs GPR:$rd), (ins uimm5:$imm, GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (load (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 (shl (i64 GPR:$rs2), (i64 (i64 uimm5:$imm)))))))))),
	(LDR_ uimm5:$imm, GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def STR_ : RVInst_STR<(outs ), (ins GPR:$rs3, GPR:$rs2, GPR:$rs1, uimm5:$imm)>;

def : Pat<
	(store (XLenVT GPR:$rs3), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 (shl (i64 GPR:$rs2), (i64 (i64 uimm5:$imm)))))))),
	(STR_ GPR:$rs3, GPR:$rs2, GPR:$rs1, uimm5:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def ADD3_ : RVInst_ADD3<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i64 (add (add GPR:$rs2, GPR:$rs1), GPR:$rd)),
	(ADD3_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

}
