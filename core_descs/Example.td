let Predicates = [HasVendorXCValu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_SUBINCACC_ : RVInst_CV_SUBINCACC<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i32 (add (sub (add GPR:$rs1, (i32 1)), GPR:$rs2), GPR:$rd)),
	(CV_SUBINCACC_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

let Predicates = [HasVendorXCValu], hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_MAXU_ : RVInst_CV_MAXU<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (setcc GPR:$rs1, GPR:$rs2, SETUGT))),
	(CV_MAXU_ GPR:$rs2, GPR:$rs1)>;

