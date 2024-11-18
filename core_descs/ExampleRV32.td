let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LB_ : RVInst_LB<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (sextloadi8 (add GPR:$rs1, (i32 uimm12:$imm))))),
	(LB_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LH_ : RVInst_LH<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (sextloadi16 (add GPR:$rs1, (i32 uimm12:$imm))))),
	(LH_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LW_ : RVInst_LW<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (load (add GPR:$rs1, (i32 uimm12:$imm))))),
	(LW_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LBU_ : RVInst_LBU<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (zextloadi8 (add GPR:$rs1, (i32 uimm12:$imm))))),
	(LBU_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LHU_ : RVInst_LHU<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (zextloadi16 (add GPR:$rs1, (i32 uimm12:$imm))))),
	(LHU_ uimm12:$imm, GPR:$rs1)>;

}
