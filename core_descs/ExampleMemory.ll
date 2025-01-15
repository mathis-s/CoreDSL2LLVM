; ModuleID = 'mod'
source_filename = "mod"

define void @implLDR(i64 %imm, ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 31
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs2.v = load i64, ptr %rs2, align 8
  %3 = shl i64 %rs2.v, %imm
  %rs1.v = load i64, ptr %rs1, align 8
  %4 = add i64 %rs1.v, %3
  %5 = alloca i64, align 8
  store i64 %4, ptr %5, align 4
  %.v = load i64, ptr %5, align 8
  %6 = inttoptr i64 %.v to ptr
  %.v1 = load i64, ptr %6, align 8
  %7 = alloca i64, align 8
  store i64 %.v1, ptr %7, align 4
  br i1 true, label %8, label %9

8:                                                ; preds = %0
  %.v2 = load i64, ptr %7, align 8
  store i64 %.v2, ptr %rd, align 8
  br label %9

9:                                                ; preds = %8, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implSTR(ptr %rs3, ptr %rs2, ptr %rs1, i64 %imm) {
  %1 = and i64 %imm, 31
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs2.v = load i64, ptr %rs2, align 8
  %3 = shl i64 %rs2.v, %imm
  %rs1.v = load i64, ptr %rs1, align 8
  %4 = add i64 %rs1.v, %3
  %5 = alloca i64, align 8
  store i64 %4, ptr %5, align 4
  %.v = load i64, ptr %5, align 8
  %6 = inttoptr i64 %.v to ptr
  %rs3.v = load i64, ptr %rs3, align 8
  store i64 %rs3.v, ptr %6, align 8
  ret void
}

define void @implLOADMAC(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i64, ptr %rs1, align 8
  %1 = inttoptr i64 %rs1.v to ptr
  %rs2.v = load i64, ptr %rs2, align 8
  %2 = inttoptr i64 %rs2.v to ptr
  %.v = load i64, ptr %1, align 8
  %.v1 = load i64, ptr %2, align 8
  %3 = mul i64 %.v, %.v1
  %rd.v = load i64, ptr %rd, align 8
  %4 = add i64 %rd.v, %3
  store i64 %4, ptr %rd, align 8
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

