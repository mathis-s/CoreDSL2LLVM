; ModuleID = 'mod'
source_filename = "mod"

define void @implFMAC(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %rd.v = load i32, ptr %rd, align 4
  %1 = bitcast i32 %rs1.v to float
  %2 = bitcast i32 %rs2.v to float
  %3 = bitcast i32 %rd.v to float
  %4 = call float @llvm.fmuladd.f32(float %3, float %1, float %2)
  %5 = bitcast float %4 to i32
  store i32 %5, ptr %rd, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #0

define void @implFMEAN(ptr noalias %rd, ptr %rs1, ptr %rs2) {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %1 = bitcast i32 %rs1.v to float
  %2 = bitcast i32 %rs2.v to float
  %3 = fadd float %1, %2
  %4 = bitcast float %3 to i32
  %5 = bitcast i32 %4 to float
  %6 = fdiv float %5, 2.000000e+00
  %7 = bitcast float %6 to i32
  store i32 %7, ptr %rd, align 4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }

