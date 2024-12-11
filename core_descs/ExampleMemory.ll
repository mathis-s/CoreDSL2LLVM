; ModuleID = 'mod'
source_filename = "mod"

define void @implLDR(i64 %imm, ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 31
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs2.v = load i64, ptr %rs2, align 8
  %3 = shl i64 %rs2.v, %imm
  %rs1.v = load i64, ptr %rs1, align 8
  %4 = zext i64 %rs1.v to i128
  %5 = zext i64 %3 to i128
  %6 = add i128 %4, %5
  %7 = trunc i128 %6 to i64
  %8 = alloca i64, align 8
  store i64 %7, ptr %8, align 4
  %.v = load i64, ptr %8, align 8
  %9 = inttoptr i64 %.v to ptr
  %.v1 = load i64, ptr %9, align 8
  %10 = alloca i64, align 8
  store i64 %.v1, ptr %10, align 4
  br i1 true, label %11, label %12

11:                                               ; preds = %0
  %.v2 = load i64, ptr %10, align 8
  store i64 %.v2, ptr %rd, align 8
  br label %12

12:                                               ; preds = %11, %0
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
  %4 = zext i64 %rs1.v to i128
  %5 = zext i64 %3 to i128
  %6 = add i128 %4, %5
  %7 = trunc i128 %6 to i64
  %8 = alloca i64, align 8
  store i64 %7, ptr %8, align 4
  %.v = load i64, ptr %8, align 8
  %9 = inttoptr i64 %.v to ptr
  %rs3.v = load i64, ptr %rs3, align 8
  store i64 %rs3.v, ptr %9, align 8
  ret void
}

define void @implADD3(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %1 = zext i64 %rs1.v to i128
  %2 = zext i64 %rs2.v to i128
  %3 = add i128 %1, %2
  %rd.v = load i64, ptr %rd, align 8
  %4 = zext i64 %rd.v to i256
  %5 = zext i128 %3 to i256
  %6 = add i256 %4, %5
  %7 = trunc i256 %6 to i64
  store i64 %7, ptr %rd, align 8
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

