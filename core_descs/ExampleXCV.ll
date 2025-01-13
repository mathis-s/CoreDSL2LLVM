; ModuleID = 'mod'
source_filename = "mod"

define void @implCV_MAC(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %1 = mul i32 %rs1.v, %rs2.v
  %rd.v = load i32, ptr %rd, align 4
  %2 = add i32 %1, %rd.v
  %3 = sext i32 %2 to i128
  %4 = alloca i128, align 8
  store i128 %3, ptr %4, align 4
  br i1 true, label %5, label %7

5:                                                ; preds = %0
  %.v = load i65, ptr %4, align 16
  %6 = trunc i65 %.v to i32
  store i32 %6, ptr %rd, align 4
  br label %7

7:                                                ; preds = %5, %0
  ret void
}

define void @implCV_ABS(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %10

1:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %2 = icmp slt i32 %rs1.v, 0
  %rs1.v1 = load i32, ptr %rs1, align 4
  %3 = zext i32 %rs1.v1 to i64
  %4 = sub i64 0, %3
  %5 = lshr i64 %4, 0
  %6 = and i64 %5, 4294967295
  %7 = trunc i64 %6 to i32
  %rs1.v2 = load i32, ptr %rs1, align 4
  %8 = icmp ne i1 %2, false
  %9 = select i1 %8, i32 %7, i32 %rs1.v2
  store i32 %9, ptr %rd, align 4
  br label %10

10:                                               ; preds = %1, %0
  ret void
}

define void @implCV_ADDN(i32 %Luimm5, ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %Luimm5, 31
  %2 = icmp eq i32 %Luimm5, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %6

3:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %4 = add i32 %rs1.v, %rs2.v
  %5 = ashr i32 %4, %Luimm5
  store i32 %5, ptr %rd, align 4
  br label %6

6:                                                ; preds = %3, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implCV_ADD_H(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %14

1:                                                ; preds = %0
  %2 = getelementptr i16, ptr %rd, i32 0
  %3 = getelementptr i16, ptr %rs1, i32 0
  %4 = getelementptr i16, ptr %rs2, i32 0
  %.v = load i16, ptr %3, align 2
  %.v1 = load i16, ptr %4, align 2
  %5 = add i16 %.v, %.v1
  %6 = lshr i16 %5, 0
  %7 = and i16 %6, -1
  store i16 %7, ptr %2, align 2
  %8 = getelementptr i16, ptr %rd, i32 1
  %9 = getelementptr i16, ptr %rs1, i32 1
  %10 = getelementptr i16, ptr %rs2, i32 1
  %.v2 = load i16, ptr %9, align 2
  %.v3 = load i16, ptr %10, align 2
  %11 = add i16 %.v2, %.v3
  %12 = lshr i16 %11, 0
  %13 = and i16 %12, -1
  store i16 %13, ptr %8, align 2
  br label %14

14:                                               ; preds = %1, %0
  ret void
}

define void @implCV_ADD_SC_H(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %14

1:                                                ; preds = %0
  %2 = getelementptr i16, ptr %rd, i32 0
  %3 = getelementptr i16, ptr %rs1, i32 0
  %4 = getelementptr i16, ptr %rs2, i32 0
  %.v = load i16, ptr %3, align 2
  %.v1 = load i16, ptr %4, align 2
  %5 = add i16 %.v, %.v1
  %6 = lshr i16 %5, 0
  %7 = and i16 %6, -1
  store i16 %7, ptr %2, align 2
  %8 = getelementptr i16, ptr %rd, i32 1
  %9 = getelementptr i16, ptr %rs1, i32 1
  %10 = getelementptr i16, ptr %rs2, i32 0
  %.v2 = load i16, ptr %9, align 2
  %.v3 = load i16, ptr %10, align 2
  %11 = add i16 %.v2, %.v3
  %12 = lshr i16 %11, 0
  %13 = and i16 %12, -1
  store i16 %13, ptr %8, align 2
  br label %14

14:                                               ; preds = %1, %0
  ret void
}

define void @implCV_ADD_SCI_H(i32 %Imm6, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %Imm6, 63
  %2 = icmp eq i32 %Imm6, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %16

3:                                                ; preds = %0
  %4 = getelementptr i16, ptr %rd, i32 0
  %5 = getelementptr i16, ptr %rs1, i32 0
  %6 = trunc i32 %Imm6 to i16
  %.v = load i16, ptr %5, align 2
  %7 = add i16 %.v, %6
  %8 = lshr i16 %7, 0
  %9 = and i16 %8, -1
  store i16 %9, ptr %4, align 2
  %10 = getelementptr i16, ptr %rd, i32 1
  %11 = getelementptr i16, ptr %rs1, i32 1
  %12 = trunc i32 %Imm6 to i16
  %.v1 = load i16, ptr %11, align 2
  %13 = add i16 %.v1, %12
  %14 = lshr i16 %13, 0
  %15 = and i16 %14, -1
  store i16 %15, ptr %10, align 2
  br label %16

16:                                               ; preds = %3, %0
  ret void
}

define void @implCV_ADD_B(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %26

1:                                                ; preds = %0
  %2 = getelementptr i8, ptr %rd, i32 0
  %3 = getelementptr i8, ptr %rs1, i32 0
  %4 = getelementptr i8, ptr %rs2, i32 0
  %.v = load i8, ptr %3, align 1
  %.v1 = load i8, ptr %4, align 1
  %5 = add i8 %.v, %.v1
  %6 = lshr i8 %5, 0
  %7 = and i8 %6, -1
  store i8 %7, ptr %2, align 1
  %8 = getelementptr i8, ptr %rd, i32 1
  %9 = getelementptr i8, ptr %rs1, i32 1
  %10 = getelementptr i8, ptr %rs2, i32 1
  %.v2 = load i8, ptr %9, align 1
  %.v3 = load i8, ptr %10, align 1
  %11 = add i8 %.v2, %.v3
  %12 = lshr i8 %11, 0
  %13 = and i8 %12, -1
  store i8 %13, ptr %8, align 1
  %14 = getelementptr i8, ptr %rd, i32 2
  %15 = getelementptr i8, ptr %rs1, i32 2
  %16 = getelementptr i8, ptr %rs2, i32 2
  %.v4 = load i8, ptr %15, align 1
  %.v5 = load i8, ptr %16, align 1
  %17 = add i8 %.v4, %.v5
  %18 = lshr i8 %17, 0
  %19 = and i8 %18, -1
  store i8 %19, ptr %14, align 1
  %20 = getelementptr i8, ptr %rd, i32 3
  %21 = getelementptr i8, ptr %rs1, i32 3
  %22 = getelementptr i8, ptr %rs2, i32 3
  %.v6 = load i8, ptr %21, align 1
  %.v7 = load i8, ptr %22, align 1
  %23 = add i8 %.v6, %.v7
  %24 = lshr i8 %23, 0
  %25 = and i8 %24, -1
  store i8 %25, ptr %20, align 1
  br label %26

26:                                               ; preds = %1, %0
  ret void
}

define void @implCV_ADD_SC_B(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %26

1:                                                ; preds = %0
  %2 = getelementptr i8, ptr %rd, i32 0
  %3 = getelementptr i8, ptr %rs1, i32 0
  %4 = getelementptr i8, ptr %rs2, i32 0
  %.v = load i8, ptr %3, align 1
  %.v1 = load i8, ptr %4, align 1
  %5 = add i8 %.v, %.v1
  %6 = lshr i8 %5, 0
  %7 = and i8 %6, -1
  store i8 %7, ptr %2, align 1
  %8 = getelementptr i8, ptr %rd, i32 1
  %9 = getelementptr i8, ptr %rs1, i32 1
  %10 = getelementptr i8, ptr %rs2, i32 0
  %.v2 = load i8, ptr %9, align 1
  %.v3 = load i8, ptr %10, align 1
  %11 = add i8 %.v2, %.v3
  %12 = lshr i8 %11, 0
  %13 = and i8 %12, -1
  store i8 %13, ptr %8, align 1
  %14 = getelementptr i8, ptr %rd, i32 2
  %15 = getelementptr i8, ptr %rs1, i32 2
  %16 = getelementptr i8, ptr %rs2, i32 0
  %.v4 = load i8, ptr %15, align 1
  %.v5 = load i8, ptr %16, align 1
  %17 = add i8 %.v4, %.v5
  %18 = lshr i8 %17, 0
  %19 = and i8 %18, -1
  store i8 %19, ptr %14, align 1
  %20 = getelementptr i8, ptr %rd, i32 3
  %21 = getelementptr i8, ptr %rs1, i32 3
  %22 = getelementptr i8, ptr %rs2, i32 0
  %.v6 = load i8, ptr %21, align 1
  %.v7 = load i8, ptr %22, align 1
  %23 = add i8 %.v6, %.v7
  %24 = lshr i8 %23, 0
  %25 = and i8 %24, -1
  store i8 %25, ptr %20, align 1
  br label %26

26:                                               ; preds = %1, %0
  ret void
}

define void @implCV_ADD_SCI_B(i32 %Imm6, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %Imm6, 63
  %2 = icmp eq i32 %Imm6, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %28

3:                                                ; preds = %0
  %4 = getelementptr i8, ptr %rd, i32 0
  %5 = getelementptr i8, ptr %rs1, i32 0
  %6 = trunc i32 %Imm6 to i8
  %.v = load i8, ptr %5, align 1
  %7 = add i8 %.v, %6
  %8 = lshr i8 %7, 0
  %9 = and i8 %8, -1
  store i8 %9, ptr %4, align 1
  %10 = getelementptr i8, ptr %rd, i32 1
  %11 = getelementptr i8, ptr %rs1, i32 1
  %12 = trunc i32 %Imm6 to i8
  %.v1 = load i8, ptr %11, align 1
  %13 = add i8 %.v1, %12
  %14 = lshr i8 %13, 0
  %15 = and i8 %14, -1
  store i8 %15, ptr %10, align 1
  %16 = getelementptr i8, ptr %rd, i32 2
  %17 = getelementptr i8, ptr %rs1, i32 2
  %18 = trunc i32 %Imm6 to i8
  %.v2 = load i8, ptr %17, align 1
  %19 = add i8 %.v2, %18
  %20 = lshr i8 %19, 0
  %21 = and i8 %20, -1
  store i8 %21, ptr %16, align 1
  %22 = getelementptr i8, ptr %rd, i32 3
  %23 = getelementptr i8, ptr %rs1, i32 3
  %24 = trunc i32 %Imm6 to i8
  %.v3 = load i8, ptr %23, align 1
  %25 = add i8 %.v3, %24
  %26 = lshr i8 %25, 0
  %27 = and i8 %26, -1
  store i8 %27, ptr %22, align 1
  br label %28

28:                                               ; preds = %3, %0
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

