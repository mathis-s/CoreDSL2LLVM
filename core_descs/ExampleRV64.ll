; ModuleID = 'mod'
source_filename = "mod"

define void @implADDW2(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = getelementptr i32, ptr %rs1, i64 0
  %.v = load i32, ptr %1, align 4
  %2 = getelementptr i32, ptr %rs2, i64 0
  %.v1 = load i32, ptr %2, align 4
  %3 = add i32 %.v, %.v1
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v2 = load i32, ptr %4, align 4
  %5 = sext i32 %.v2 to i64
  store i64 %5, ptr %rd, align 8
  ret void
}

define void @implADDW3(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %1 = add i32 %rs1.v, %rs2.v
  %2 = alloca i32, align 4
  store i32 %1, ptr %2, align 4
  %.v = load i32, ptr %2, align 4
  %3 = sext i32 %.v to i64
  store i64 %3, ptr %rd, align 8
  ret void
}

define void @implLB(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i8, ptr %5, align 1
  %6 = alloca i8, align 1
  store i8 %.v1, ptr %6, align 1
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i8, ptr %6, align 1
  %8 = sext i8 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implLH(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i16, ptr %5, align 2
  %6 = alloca i16, align 2
  store i16 %.v1, ptr %6, align 2
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i16, ptr %6, align 2
  %8 = sext i16 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLW(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i32, ptr %5, align 4
  %6 = alloca i32, align 4
  store i32 %.v1, ptr %6, align 4
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i32, ptr %6, align 4
  %8 = sext i32 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLD(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i64, ptr %5, align 8
  %6 = alloca i64, align 8
  store i64 %.v1, ptr %6, align 4
  br i1 true, label %7, label %8

7:                                                ; preds = %0
  %.v2 = load i64, ptr %6, align 8
  store i64 %.v2, ptr %rd, align 8
  br label %8

8:                                                ; preds = %7, %0
  ret void
}

define void @implLBU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i8, ptr %5, align 1
  %6 = alloca i8, align 1
  store i8 %.v1, ptr %6, align 1
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i8, ptr %6, align 1
  %8 = zext i8 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLHU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i16, ptr %5, align 2
  %6 = alloca i16, align 2
  store i16 %.v1, ptr %6, align 2
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i16, ptr %6, align 2
  %8 = zext i16 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLWU(i64 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i64 %imm, 4095
  %2 = icmp eq i64 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i64, ptr %rs1, align 8
  %3 = add i64 %rs1.v, %imm
  %4 = alloca i64, align 8
  store i64 %3, ptr %4, align 4
  %.v = load i64, ptr %4, align 8
  %5 = inttoptr i64 %.v to ptr
  %.v1 = load i32, ptr %5, align 4
  %6 = alloca i32, align 4
  store i32 %.v1, ptr %6, align 4
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i32, ptr %6, align 4
  %8 = zext i32 %.v2 to i64
  store i64 %8, ptr %rd, align 8
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

