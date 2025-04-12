let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd = $rd_wb" in def MAC_LD_X2_n2_ : RVInst_MAC_LD_X2_n2<(outs GPR:$rd2_wb, GPR:$rd_wb), (ins GPR:$rd2, GPR:$rs2, GPR:$rs1, GPR:$rd)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd = $rd_wb" in def MAC_LD_X2_n1_ : RVInst_MAC_LD_X2_n1<(outs GPR:$rd2_wb, GPR:$rd_wb), (ins GPR:$rd2, GPR:$rs2, GPR:$rs1, GPR:$rd)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd = $rd_wb" in def MAC_LD_X2_0_ : RVInst_MAC_LD_X2_0<(outs GPR:$rd2_wb, GPR:$rd_wb), (ins GPR:$rd2, GPR:$rs2, GPR:$rs1, GPR:$rd)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isBranch = 0, isTerminator = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd = $rd_wb" in def MAC_LD_X2_1_ : RVInst_MAC_LD_X2_1<(outs GPR:$rd2_wb, GPR:$rd_wb), (ins GPR:$rd2, GPR:$rs2, GPR:$rs1, GPR:$rd)>;
}
