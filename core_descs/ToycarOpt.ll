; ModuleID = 'mod'
source_filename = "mod"

define internal void @implMAC_LD_X2(i32 %imm2, ptr noalias %rd2, ptr %rs2, ptr %rs1, ptr noalias %rd) {
  %1 = shl i32 %imm2, 30
  %2 = ashr i32 %1, 30
  %3 = icmp eq i32 %imm2, %2
  call void @llvm.assume(i1 %3)
  %4 = sext i32 %imm2 to i35
  %5 = mul i35 %4, 4
  %rs2.v = load i32, ptr %rs2, align 4
  %6 = zext i32 %rs2.v to i36
  %7 = sext i35 %5 to i36
  %8 = add i36 %6, %7
  %9 = inttoptr i36 %8 to ptr
  %.v = load i16, ptr %9, align 2
  %rs1.v = load i32, ptr %rs1, align 4
  %10 = zext i32 %rs1.v to i48
  %11 = sext i16 %.v to i48
  %12 = mul i48 %10, %11
  %rd.v = load i32, ptr %rd, align 4
  %13 = zext i32 %rd.v to i49
  %14 = sext i48 %12 to i49
  %15 = add i49 %13, %14
  %16 = trunc i49 %15 to i32
  store i32 %16, ptr %rd, align 4
  %17 = sext i32 %imm2 to i35
  %18 = mul i35 %17, 4
  %rs2.v1 = load i32, ptr %rs2, align 4
  %19 = zext i32 %rs2.v1 to i36
  %20 = sext i35 %18 to i36
  %21 = add i36 %19, %20
  %22 = sext i36 %21 to i37
  %23 = add i37 %22, 2
  %24 = inttoptr i37 %23 to ptr
  %.v2 = load i16, ptr %24, align 2
  %rs1.v3 = load i32, ptr %rs1, align 4
  %25 = zext i32 %rs1.v3 to i48
  %26 = sext i16 %.v2 to i48
  %27 = mul i48 %25, %26
  %rd2.v = load i32, ptr %rd2, align 4
  %28 = zext i32 %rd2.v to i49
  %29 = sext i48 %27 to i49
  %30 = add i49 %28, %29
  %31 = trunc i49 %30 to i32
  store i32 %31, ptr %rd2, align 4
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implMAC_LD_X2_n2(ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd) {
  call void @implMAC_LD_X2(i32 -2, ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd)
  ret void
}

define void @implMAC_LD_X2_n1(ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd) {
  call void @implMAC_LD_X2(i32 -1, ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd)
  ret void
}

define void @implMAC_LD_X2_0(ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd) {
  call void @implMAC_LD_X2(i32 0, ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd)
  ret void
}

define void @implMAC_LD_X2_1(ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd) {
  call void @implMAC_LD_X2(i32 1, ptr %rd2, ptr %rs2, ptr %rs1, ptr %rd)
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

