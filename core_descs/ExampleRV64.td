let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def ADDW2_ : RVInst_ADDW2<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sext (i32 (add (i32 GPR:$rs2), (i32 GPR:$rs1)))))),
	(ADDW2_ GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def ADDW3_ : RVInst_ADDW3<(outs GPR:$rd), (ins GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i64 (i64 (sext (i32 (add (i32 GPR:$rs2), (i32 GPR:$rs1)))))),
	(ADDW3_ GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LB_ : RVInst_LB<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (sextloadi8 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LB_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LH_ : RVInst_LH<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (sextloadi16 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LH_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LW_ : RVInst_LW<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (sextloadi32 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LW_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LD_ : RVInst_LD<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (load (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LD_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LBU_ : RVInst_LBU<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (zextloadi8 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LBU_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LHU_ : RVInst_LHU<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (zextloadi16 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LHU_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LWU_ : RVInst_LWU<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i64 (i64 (zextloadi32 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))))),
	(LWU_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SB_ : RVInst_SB<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(truncstorei8 (XLenVT (i32 (trunc GPR:$rs2))), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))),
	(SB_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SH_ : RVInst_SH<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(truncstorei16 (XLenVT (i32 (trunc GPR:$rs2))), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))),
	(SH_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SW_ : RVInst_SW<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(truncstorei32 (XLenVT (i32 (trunc GPR:$rs2))), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i64 (i64 simm12:$imm))))),
	(SW_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

}
