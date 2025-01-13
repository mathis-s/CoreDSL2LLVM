; ModuleID = 'mod'
source_filename = "mod"

define void @implFMAC(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %rd.v = load i64, ptr %rd, align 8
  %1 = bitcast i64 %rs1.v to double
  %2 = bitcast i64 %rs2.v to double
  %3 = bitcast i64 %rd.v to double
  %4 = call double @llvm.fmuladd.f64(double %3, double %1, double %2)
  %5 = bitcast double %4 to i64
  store i64 %5, ptr %rd, align 8
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #0

define void @implFMEAN(ptr noalias %rd, ptr %rs1, ptr %rs2) {
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %1 = bitcast i64 %rs1.v to double
  %2 = bitcast i64 %rs2.v to double
  %3 = fadd double %1, %2
  %4 = bitcast double %3 to i64
  %5 = bitcast i64 %4 to double
  %6 = fdiv double %5, 2.000000e+00
  %7 = bitcast double %6 to i64
  store i64 %7, ptr %rd, align 8
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

