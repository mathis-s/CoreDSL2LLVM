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

