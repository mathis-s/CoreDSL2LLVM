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

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

