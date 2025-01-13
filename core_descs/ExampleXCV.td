let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd = $rd_wb" in def CV_MAC_ : RVInst_CV_MAC<(outs GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;

def : Pat<
	(i32 (add (mul GPR:$rs2, GPR:$rs1), GPR:$rd)),
	(CV_MAC_ GPR:$rs2, GPR:$rs1, GPR:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_ABS_ : RVInst_CV_ABS<(outs GPR:$rd), (ins GPR:$rs1)>;

def : Pat<
	(i32 (xor (add GPR:$rs1, (i32 (sra (i32 GPR:$rs1), (i32 (i32 31))))), (i32 (sra (i32 GPR:$rs1), (i32 (i32 31)))))),
	(CV_ABS_ GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def CV_ADDN_ : RVInst_CV_ADDN<(outs GPR:$rd), (ins uimm5:$Luimm5, GPR:$rs2, GPR:$rs1)>;

def : Pat<
	(i32 (i32 (sra (i32 (add GPR:$rs2, GPR:$rs1)), (i32 (i32 uimm5:$Luimm5))))),
	(CV_ADDN_ uimm5:$Luimm5, GPR:$rs2, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_H_ : RVInst_CV_ADD_H<(outs ), (ins GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei16 (XLenVT (add (i32 (srl GPR:$rs2, (i32 16))), (i32 (srl GPR:$rs1, (i32 16))))), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 2))))),
	(CV_ADD_H_ GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_SC_H_ : RVInst_CV_ADD_SC_H<(outs ), (ins GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei16 (XLenVT (add (i32 (srl GPR:$rs1, (i32 16))), GPR:$rs2)), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 2))))),
	(CV_ADD_SC_H_ GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_SCI_H_ : RVInst_CV_ADD_SCI_H<(outs ), (ins uimm6:$Imm6, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei16 (XLenVT (add (i32 (srl GPR:$rs1, (i32 16))), (i32 uimm6:$Imm6))), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 2))))),
	(CV_ADD_SCI_H_ uimm6:$Imm6, GPR:$rs1, uimm5:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_B_ : RVInst_CV_ADD_B<(outs ), (ins GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei8 (XLenVT (add (i32 (srl GPR:$rs2, (i32 24))), (i32 (srl GPR:$rs1, (i32 24))))), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 3))))),
	(CV_ADD_B_ GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_SC_B_ : RVInst_CV_ADD_SC_B<(outs ), (ins GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei8 (XLenVT (add (i32 (srl GPR:$rs1, (i32 24))), GPR:$rs2)), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 3))))),
	(CV_ADD_SC_B_ GPR:$rs2, GPR:$rs1, uimm5:$rd)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 1, isCodeGenOnly = 1, Constraints = "" in def CV_ADD_SCI_B_ : RVInst_CV_ADD_SCI_B<(outs ), (ins uimm6:$Imm6, GPR:$rs1, uimm5:$rd)>;

def : Pat<
	(truncstorei8 (XLenVT (add (i32 (srl GPR:$rs1, (i32 24))), (i32 uimm6:$Imm6))), (iPTR (ptradd (iPTR (i32 uimm5:$rd)), (i32 (i32 3))))),
	(CV_ADD_SCI_B_ uimm6:$Imm6, GPR:$rs1, uimm5:$rd)>;

}
