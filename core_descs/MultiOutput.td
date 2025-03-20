let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd1 = $rd1_wb" in def DUALADD_ : RVInst_DUALADD<(outs GPR:$rd2_wb, GPR:$rd1_wb), (ins GPR:$rs1, GPR:$rd2, GPR:$rd1)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb" in def LOAD_ACC_ : RVInst_LOAD_ACC<(outs GPR:$rd2_wb, GPR:$rd1), (ins GPR:$rs1, GPR:$rd2)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rs2 = $rs2_wb, $rs1 = $rs1_wb, $rd = $rd_wb" in def DOTP_ : RVInst_DOTP<(outs GPR:$rs2_wb, GPR:$rs1_wb, GPR:$rd_wb), (ins GPR:$rs2, GPR:$rs1, GPR:$rd)>;
let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "" in def UNPACK16_ : RVInst_UNPACK16<(outs GPR:$rd2, GPR:$rd1), (ins GPR:$rs1)>;
}
