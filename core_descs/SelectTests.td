let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def SEXT_BIT23_ : RVInst_SEXT_BIT23<(outs GPR:$rd), (ins GPR:$rs1)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def ZEXT_BIT23_ : RVInst_ZEXT_BIT23<(outs GPR:$rd), (ins GPR:$rs1)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def ZEXT_UNALIGNED_BYTE_ : RVInst_ZEXT_UNALIGNED_BYTE<(outs GPR:$rd), (ins GPR:$rs1)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def SEXT_UNALIGNED_BYTE_ : RVInst_SEXT_UNALIGNED_BYTE<(outs GPR:$rd), (ins GPR:$rs1)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def ZEXT_ALIGNED_WORD_ : RVInst_ZEXT_ALIGNED_WORD<(outs GPR:$rd), (ins GPR:$rs1)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "" in def SEXT_ALIGNED_WORD_ : RVInst_SEXT_ALIGNED_WORD<(outs GPR:$rd), (ins GPR:$rs1)>;
}
