; ModuleID = 'mod'
source_filename = "mod"

define void @implCV_SUBINCACC(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %5

1:                                                ; preds = %0
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %2 = sub i64 %rs1.v, %rs2.v
  %3 = add i64 %2, 1
  %rd.v = load i64, ptr %rd, align 8
  %4 = add i64 %rd.v, %3
  store i64 %4, ptr %rd, align 8
  br label %5

5:                                                ; preds = %1, %0
  ret void
}

define void @implCV_MAXU(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %5

1:                                                ; preds = %0
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %2 = icmp ugt i64 %rs1.v, %rs2.v
  %rs1.v1 = load i64, ptr %rs1, align 8
  %rs2.v2 = load i64, ptr %rs2, align 8
  %3 = icmp ne i1 %2, false
  %4 = select i1 %3, i64 %rs1.v1, i64 %rs2.v2
  store i64 %4, ptr %rd, align 8
  br label %5

5:                                                ; preds = %1, %0
  ret void
}

define void @implNAND(ptr %rs2, ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %4

1:                                                ; preds = %0
  %rs1.v = load i64, ptr %rs1, align 8
  %rs2.v = load i64, ptr %rs2, align 8
  %2 = and i64 %rs1.v, %rs2.v
  %3 = xor i64 %2, -1
  store i64 %3, ptr %rd, align 8
  br label %4

4:                                                ; preds = %1, %0
  ret void
}

