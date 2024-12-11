; ModuleID = 'mod'
source_filename = "mod"

define void @implADDW2(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = getelementptr i32, ptr %rs1, i64 0
  %.v = load i32, ptr %1, align 4
  %2 = getelementptr i32, ptr %rs2, i64 0
  %.v1 = load i32, ptr %2, align 4
  %3 = sext i32 %.v to i64
  %4 = sext i32 %.v1 to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v2 = load i32, ptr %7, align 4
  %8 = sext i32 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  ret void
}

define void @implADDW3(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %1 = sext i32 %rs1.v to i64
  %2 = sext i32 %rs2.v to i64
  %3 = add i64 %1, %2
  %4 = trunc i64 %3 to i32
  %5 = alloca i32, align 4
  store i32 %4, ptr %5, align 4
  %.v = load i32, ptr %5, align 4
  %6 = sext i32 %.v to i64
  store i64 %6, ptr %rd, align 8
  ret void
}

define void @implLB(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i8, ptr %8, align 1
  %9 = alloca i8, align 1
  store i8 %.v1, ptr %9, align 1
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i8, ptr %9, align 1
  %11 = sext i8 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implLH(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i16, ptr %8, align 2
  %9 = alloca i16, align 2
  store i16 %.v1, ptr %9, align 2
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i16, ptr %9, align 2
  %11 = sext i16 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLW(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i32, ptr %8, align 4
  %9 = alloca i32, align 4
  store i32 %.v1, ptr %9, align 4
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i32, ptr %9, align 4
  %11 = sext i32 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLD(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i64, ptr %8, align 8
  %9 = alloca i64, align 8
  store i64 %.v1, ptr %9, align 4
  br i1 true, label %10, label %11

10:                                               ; preds = %0
  %.v2 = load i64, ptr %9, align 8
  store i64 %.v2, ptr %rd, align 8
  br label %11

11:                                               ; preds = %10, %0
  ret void
}

define void @implLBU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i8, ptr %8, align 1
  %9 = alloca i8, align 1
  store i8 %.v1, ptr %9, align 1
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i8, ptr %9, align 1
  %11 = zext i8 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLHU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i16, ptr %8, align 2
  %9 = alloca i16, align 2
  store i16 %.v1, ptr %9, align 2
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i16, ptr %9, align 2
  %11 = zext i16 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLWU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %.v1 = load i32, ptr %8, align 4
  %9 = alloca i32, align 4
  store i32 %.v1, ptr %9, align 4
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i32, ptr %9, align 4
  %11 = zext i32 %.v2 to i64
  store i64 %11, ptr %rd, align 8
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implSB(i64 %imm, ptr %rs2, ptr %rs1) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %rs2.v = load i64, ptr %rs2, align 8
  %9 = trunc i64 %rs2.v to i8
  store i8 %9, ptr %8, align 1
  ret void
}

define void @implSH(i64 %imm, ptr %rs2, ptr %rs1) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %rs2.v = load i64, ptr %rs2, align 8
  %9 = trunc i64 %rs2.v to i16
  store i16 %9, ptr %8, align 2
  ret void
}

define void @implSW(i64 %imm, ptr %rs2, ptr %rs1) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = zext i64 %rs1.v to i128
  %4 = sext i64 %imm to i128
  %5 = add i128 %3, %4
  %6 = trunc i128 %5 to i64
  %7 = alloca i64, align 8
  store i64 %6, ptr %7, align 4
  %.v = load i64, ptr %7, align 8
  %8 = inttoptr i64 %.v to ptr
  %rs2.v = load i64, ptr %rs2, align 8
  %9 = trunc i64 %rs2.v to i32
  store i32 %9, ptr %8, align 4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

