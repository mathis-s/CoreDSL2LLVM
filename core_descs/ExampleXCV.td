let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_MAC_ : RVInst_CV_MAC<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i32 (add (mul GPR:$rs2, GPR:$rs1), GPR:$rd)),
	(CV_MAC_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_ABS_ : RVInst_CV_ABS<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(i32 (xor (add GPR:$rs1, (i32 (sra (i32 GPR:$rs1), (i32 (i32 31))))), (i32 (sra (i32 GPR:$rs1), (i32 (i32 31)))))),
	(CV_ABS_ GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_ADDN_ : RVInst_CV_ADDN<(outs GPR:$rd), (ins uimm5:$Luimm5, GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (sra (i32 (add GPR:$rs2, GPR:$rs1)), (i32 (i32 uimm5:$Luimm5))))),
	(CV_ADDN_ uimm5:$Luimm5, GPR:$rs2, GPR:$rs1)>;

}
