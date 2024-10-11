let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def ADDW2_ : RVInst_ADDW2<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sext (i32 (add (i32 GPR:$rs2), (i32 GPR:$rs1)))))),
	(ADDW2_ GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def ADDW3_ : RVInst_ADDW3<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sext (i32 (add (i32 GPR:$rs2), (i32 GPR:$rs1)))))),
	(ADDW3_ GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LB_ : RVInst_LB<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sextloadi8 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LB_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LH_ : RVInst_LH<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sextloadi16 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LH_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LW_ : RVInst_LW<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sextloadi32 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LW_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LD_ : RVInst_LD<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (load (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LD_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LBU_ : RVInst_LBU<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (zextloadi8 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LBU_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LHU_ : RVInst_LHU<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (zextloadi16 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LHU_ uimm12:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LWU_ : RVInst_LWU<(outs GPR:$rd), (ins uimm12:$imm, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (zextloadi32 (add GPR:$rs1, (i64 uimm12:$imm))))),
	(LWU_ uimm12:$imm, GPR:$rs1)>;

}
