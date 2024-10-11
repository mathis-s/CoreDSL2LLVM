; ModuleID = 'mod'
source_filename = "mod"

define void @implLB(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v to i64
  %4 = sext i32 %imm to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v = load i32, ptr %7, align 4
  %8 = inttoptr i32 %.v to ptr
  %.v1 = load i8, ptr %8, align 1
  %9 = alloca i8, align 1
  store i8 %.v1, ptr %9, align 1
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i8, ptr %9, align 1
  %11 = sext i8 %.v2 to i32
  store i32 %11, ptr %rd, align 4
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implLH(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v to i64
  %4 = sext i32 %imm to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v = load i32, ptr %7, align 4
  %8 = inttoptr i32 %.v to ptr
  %.v1 = load i16, ptr %8, align 2
  %9 = alloca i16, align 2
  store i16 %.v1, ptr %9, align 2
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i16, ptr %9, align 2
  %11 = sext i16 %.v2 to i32
  store i32 %11, ptr %rd, align 4
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLW(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v to i64
  %4 = sext i32 %imm to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v = load i32, ptr %7, align 4
  %8 = inttoptr i32 %.v to ptr
  %.v1 = load i32, ptr %8, align 4
  %9 = alloca i32, align 4
  store i32 %.v1, ptr %9, align 4
  br i1 true, label %10, label %11

10:                                               ; preds = %0
  %.v2 = load i32, ptr %9, align 4
  store i32 %.v2, ptr %rd, align 4
  br label %11

11:                                               ; preds = %10, %0
  ret void
}

define void @implLBU(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v to i64
  %4 = sext i32 %imm to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v = load i32, ptr %7, align 4
  %8 = inttoptr i32 %.v to ptr
  %.v1 = load i8, ptr %8, align 1
  %9 = alloca i8, align 1
  store i8 %.v1, ptr %9, align 1
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i8, ptr %9, align 1
  %11 = zext i8 %.v2 to i32
  store i32 %11, ptr %rd, align 4
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

define void @implLHU(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v to i64
  %4 = sext i32 %imm to i64
  %5 = add i64 %3, %4
  %6 = trunc i64 %5 to i32
  %7 = alloca i32, align 4
  store i32 %6, ptr %7, align 4
  %.v = load i32, ptr %7, align 4
  %8 = inttoptr i32 %.v to ptr
  %.v1 = load i16, ptr %8, align 2
  %9 = alloca i16, align 2
  store i16 %.v1, ptr %9, align 2
  br i1 true, label %10, label %12

10:                                               ; preds = %0
  %.v2 = load i16, ptr %9, align 2
  %11 = zext i16 %.v2 to i32
  store i32 %11, ptr %rd, align 4
  br label %12

12:                                               ; preds = %10, %0
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

