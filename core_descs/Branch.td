let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 1, isTerminator = 1, isCodeGenOnly = 1, Constraints = "" in def BMASK_ : RVInst_BMASK<(outs ), (ins GPR:$rs1, GPR:$rs2, simm13_lsb0:$imm12)>;
}
