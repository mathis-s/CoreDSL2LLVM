; ModuleID = 'mod'
source_filename = "mod"

define void @implDUALADD(ptr %rs1, ptr noalias %rd2, ptr noalias %rd1) {
  %rd1.v = load i32, ptr %rd1, align 4
  %rs1.v = load i32, ptr %rs1, align 4
  %1 = zext i32 %rd1.v to i33
  %2 = zext i32 %rs1.v to i33
  %3 = add i33 %1, %2
  %4 = trunc i33 %3 to i32
  store i32 %4, ptr %rd1, align 4
  %rd2.v = load i32, ptr %rd2, align 4
  %rs1.v1 = load i32, ptr %rs1, align 4
  %5 = zext i32 %rd2.v to i33
  %6 = zext i32 %rs1.v1 to i33
  %7 = add i33 %5, %6
  %8 = trunc i33 %7 to i32
  store i32 %8, ptr %rd2, align 4
  ret void
}

define void @implLOAD_ACC(ptr %rs1, ptr noalias %rd2, ptr noalias %rd1) {
  %rs1.v = load i32, ptr %rs1, align 4
  %1 = inttoptr i32 %rs1.v to ptr
  %rd2.v = load i32, ptr %rd2, align 4
  %.v = load i32, ptr %1, align 4
  %2 = zext i32 %rd2.v to i33
  %3 = zext i32 %.v to i33
  %4 = add i33 %2, %3
  %5 = trunc i33 %4 to i32
  store i32 %5, ptr %rd2, align 4
  %rs1.v1 = load i32, ptr %rs1, align 4
  %6 = zext i32 %rs1.v1 to i33
  %7 = add i33 %6, 4
  %8 = trunc i33 %7 to i32
  store i32 %8, ptr %rd1, align 4
  ret void
}

define void @implDOTP(ptr noalias %rs2, ptr noalias %rs1, ptr noalias %rd) {
  %rs1.v = load i32, ptr %rs1, align 4
  %1 = inttoptr i32 %rs1.v to ptr
  %rs2.v = load i32, ptr %rs2, align 4
  %2 = inttoptr i32 %rs2.v to ptr
  %.v = load i32, ptr %1, align 4
  %.v1 = load i32, ptr %2, align 4
  %3 = zext i32 %.v to i64
  %4 = zext i32 %.v1 to i64
  %5 = mul i64 %3, %4
  %rd.v = load i32, ptr %rd, align 4
  %6 = zext i32 %rd.v to i65
  %7 = zext i64 %5 to i65
  %8 = add i65 %6, %7
  %9 = trunc i65 %8 to i32
  store i32 %9, ptr %rd, align 4
  %rs1.v2 = load i32, ptr %rs1, align 4
  %10 = zext i32 %rs1.v2 to i33
  %11 = add i33 %10, 4
  %12 = trunc i33 %11 to i32
  store i32 %12, ptr %rs1, align 4
  %rs2.v3 = load i32, ptr %rs2, align 4
  %13 = zext i32 %rs2.v3 to i33
  %14 = add i33 %13, 4
  %15 = trunc i33 %14 to i32
  store i32 %15, ptr %rs2, align 4
  ret void
}

define void @implUNPACK16(ptr %rs1, ptr noalias %rd2, ptr noalias %rd1) {
  %rs1.v = load i32, ptr %rs1, align 4
  %1 = alloca i32, align 4
  store i32 %rs1.v, ptr %1, align 4
  %2 = getelementptr i16, ptr %1, i32 1
  %.v = load i16, ptr %2, align 2
  %3 = sext i16 %.v to i32
  store i32 %3, ptr %rd1, align 4
  %4 = getelementptr i16, ptr %1, i32 0
  %.v1 = load i16, ptr %4, align 2
  %5 = sext i16 %.v1 to i32
  store i32 %5, ptr %rd2, align 4
  ret void
}

