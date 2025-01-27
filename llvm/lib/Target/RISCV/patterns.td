let Predicates = [HasVendorXCValu] in {

let hasSideEffects = 0, mayLoad = 0, mayStore = 0, isCodeGenOnly = 1, Constraints = "$rd1 = $rd1_wb, $rd2 = $rd2_wb" in def DUALADD_ : RVInst_DUALADD<(outs GPR:$rd1_wb, GPR:$rd2_wb), (ins GPR:$rs1, GPR:$rd1, GPR:$rd2)>;
}
