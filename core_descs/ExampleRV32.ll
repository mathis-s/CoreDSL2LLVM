; ModuleID = 'mod'
source_filename = "mod"

define void @implLB(ptr noalias %rd, ptr %rs1, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %.v1 = load i8, ptr %5, align 1
  %6 = alloca i8, align 1
  store i8 %.v1, ptr %6, align 1
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i8, ptr %6, align 1
  %8 = sext i8 %.v2 to i32
  store i32 %8, ptr %rd, align 4
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implLH(ptr noalias %rd, ptr %rs1, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %.v1 = load i16, ptr %5, align 2
  %6 = alloca i16, align 2
  store i16 %.v1, ptr %6, align 2
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i16, ptr %6, align 2
  %8 = sext i16 %.v2 to i32
  store i32 %8, ptr %rd, align 4
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLW(ptr noalias %rd, ptr %rs1, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %.v1 = load i32, ptr %5, align 4
  %6 = alloca i32, align 4
  store i32 %.v1, ptr %6, align 4
  br i1 true, label %7, label %8

7:                                                ; preds = %0
  %.v2 = load i32, ptr %6, align 4
  store i32 %.v2, ptr %rd, align 4
  br label %8

8:                                                ; preds = %7, %0
  ret void
}

define void @implLBU(ptr noalias %rd, ptr %rs1, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %.v1 = load i8, ptr %5, align 1
  %6 = alloca i8, align 1
  store i8 %.v1, ptr %6, align 1
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i8, ptr %6, align 1
  %8 = zext i8 %.v2 to i32
  store i32 %8, ptr %rd, align 4
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implLHU(ptr noalias %rd, ptr %rs1, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %.v1 = load i16, ptr %5, align 2
  %6 = alloca i16, align 2
  store i16 %.v1, ptr %6, align 2
  br i1 true, label %7, label %9

7:                                                ; preds = %0
  %.v2 = load i16, ptr %6, align 2
  %8 = zext i16 %.v2 to i32
  store i32 %8, ptr %rd, align 4
  br label %9

9:                                                ; preds = %7, %0
  ret void
}

define void @implSB(ptr %rs1, ptr %rs2, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %rs2.v = load i32, ptr %rs2, align 4
  %6 = trunc i32 %rs2.v to i8
  store i8 %6, ptr %5, align 1
  ret void
}

define void @implSH(ptr %rs1, ptr %rs2, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %rs2.v = load i32, ptr %rs2, align 4
  %6 = trunc i32 %rs2.v to i16
  store i16 %6, ptr %5, align 2
  ret void
}

define void @implSW(ptr %rs1, ptr %rs2, i32 %imm) {
  %1 = and i32 %imm, 4095
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %3 = add i32 %rs1.v, %imm
  %4 = alloca i32, align 4
  store i32 %3, ptr %4, align 4
  %.v = load i32, ptr %4, align 4
  %5 = inttoptr i32 %.v to ptr
  %rs2.v = load i32, ptr %rs2, align 4
  store i32 %rs2.v, ptr %5, align 4
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

