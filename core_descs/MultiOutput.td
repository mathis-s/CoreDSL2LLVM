let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb, $rd1 = $rd1_wb" in def DUALADD_ : RVInst_DUALADD<(outs GPR:$rd2_wb, GPR:$rd1_wb), (ins GPR:$rs1, GPR:$rd2, GPR:$rd1)>;
let hasSideEffects = 0, mayLoad = 1, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd2 = $rd2_wb" in def LOAD_ACC_ : RVInst_LOAD_ACC<(outs GPR:$rd2_wb, GPR:$rd1), (ins GPR:$rs1, GPR:$rd2)>;
}
