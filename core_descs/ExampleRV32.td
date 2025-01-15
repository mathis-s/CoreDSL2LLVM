let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LB_ : RVInst_LB<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i32 (i32 (sextloadi8 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))))),
	(LB_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LH_ : RVInst_LH<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i32 (i32 (sextloadi16 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))))),
	(LH_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LW_ : RVInst_LW<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i32 (i32 (load (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))))),
	(LW_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LBU_ : RVInst_LBU<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i32 (i32 (zextloadi8 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))))),
	(LBU_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def LHU_ : RVInst_LHU<(outs GPR:$rd), (ins GPR:$rs1, simm12:$imm)>;

def : Pat<
	(i32 (i32 (zextloadi16 (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))))),
	(LHU_ GPR:$rs1, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SB_ : RVInst_SB<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(truncstorei8 (XLenVT GPR:$rs2), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))),
	(SB_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SH_ : RVInst_SH<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(truncstorei16 (XLenVT GPR:$rs2), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))),
	(SH_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def SW_ : RVInst_SW<(outs ), (ins GPR:$rs1, GPR:$rs2, simm12:$imm)>;

def : Pat<
	(store (XLenVT GPR:$rs2), (iPTR (ptradd (iPTR (iPTR GPR:$rs1)), (i32 (i32 simm12:$imm))))),
	(SW_ GPR:$rs1, GPR:$rs2, simm12:$imm)>;

}
