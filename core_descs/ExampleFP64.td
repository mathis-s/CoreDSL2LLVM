let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def FMAC_ : RVInst_FMAC<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i64 (any_fma FPR64:$rd, FPR64:$rs1, FPR64:$rs2)),
	(FMAC_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def FMEAN_ : RVInst_FMEAN<(outs GPR:$rd), (ins GPR:$rs1, GPR:$rs2)>;

def : Pat<
	(i64 (fmul (fadd FPR64:$rs1, FPR64:$rs2), (f64 0.500000))),
	(FMEAN_ GPR:$rs1, GPR:$rs2)>;

}
