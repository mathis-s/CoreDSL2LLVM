let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBINCACC_ : RVInst_CV_SUBINCACC<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i64 (add (sub (add GPR:$rs1, (i64 1)), GPR:$rs2), GPR:$rd)),
	(CV_SUBINCACC_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_MAXU_ : RVInst_CV_MAXU<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (select (i64 (setcc (i64 GPR:$rs1), (i64 GPR:$rs2), SETUGT)), GPR:$rs1, GPR:$rs2)),
	(CV_MAXU_ GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def NAND_ : RVInst_NAND<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (xor (and GPR:$rs2, GPR:$rs1), (i64 -1))),
	(NAND_ GPR:$rs2, GPR:$rs1)>;

}
