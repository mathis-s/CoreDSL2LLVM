let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def K_LLI_ : RVInst_K_LLI<(outs GPR:$rd), (ins uimm32:$imm)>;

def : Pat<
	(i32 (i32 uimm32:$imm)),
	(K_LLI_ uimm32:$imm)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def K_ADDI_ : RVInst_K_ADDI<(outs GPR:$rd), (ins uimm24:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (add GPR:$rs1, (i32 uimm24:$imm))),
	(K_ADDI_ uimm24:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def K_ANDI_ : RVInst_K_ANDI<(outs GPR:$rd), (ins uimm24:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (and GPR:$rs1, (i32 uimm24:$imm))),
	(K_ANDI_ uimm24:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def K_XORI_ : RVInst_K_XORI<(outs GPR:$rd), (ins uimm24:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (xor GPR:$rs1, (i32 uimm24:$imm))),
	(K_XORI_ uimm24:$imm, GPR:$rs1)>;

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def K_ORI_ : RVInst_K_ORI<(outs GPR:$rd), (ins uimm24:$imm, GPR:$rs1)>;

def : Pat<
	(i32 (or GPR:$rs1, (i32 uimm24:$imm))),
	(K_ORI_ uimm24:$imm, GPR:$rs1)>;

}
