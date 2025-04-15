; ModuleID = 'mod'
source_filename = "mod"

define void @implSEXT_BIT23(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %6

1:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %2 = lshr i32 %rs1.v, 23
  %3 = and i32 %2, 1
  %4 = trunc i32 %3 to i1
  %5 = sext i1 %4 to i32
  store i32 %5, ptr %rd, align 4
  br label %6

6:                                                ; preds = %1, %0
  ret void
}

define void @implZEXT_BIT23(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %6

1:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %2 = lshr i32 %rs1.v, 23
  %3 = and i32 %2, 1
  %4 = trunc i32 %3 to i1
  %5 = zext i1 %4 to i32
  store i32 %5, ptr %rd, align 4
  br label %6

6:                                                ; preds = %1, %0
  ret void
}

define void @implZEXT_UNALIGNED_BYTE(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %6

1:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %2 = lshr i32 %rs1.v, 11
  %3 = and i32 %2, 255
  %4 = trunc i32 %3 to i8
  %5 = zext i8 %4 to i32
  store i32 %5, ptr %rd, align 4
  br label %6

6:                                                ; preds = %1, %0
  ret void
}

define void @implSEXT_UNALIGNED_BYTE(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %6

1:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %2 = lshr i32 %rs1.v, 11
  %3 = and i32 %2, 255
  %4 = trunc i32 %3 to i8
  %5 = sext i8 %4 to i32
  store i32 %5, ptr %rd, align 4
  br label %6

6:                                                ; preds = %1, %0
  ret void
}

define void @implZEXT_ALIGNED_WORD(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %4

1:                                                ; preds = %0
  %2 = getelementptr i16, ptr %rs1, i32 1
  %.v = load i16, ptr %2, align 2
  %3 = zext i16 %.v to i32
  store i32 %3, ptr %rd, align 4
  br label %4

4:                                                ; preds = %1, %0
  ret void
}

define void @implSEXT_ALIGNED_WORD(ptr %rs1, ptr noalias %rd) {
  br i1 true, label %1, label %4

1:                                                ; preds = %0
  %2 = getelementptr i16, ptr %rs1, i32 1
  %.v = load i16, ptr %2, align 2
  %3 = sext i16 %.v to i32
  store i32 %3, ptr %rd, align 4
  br label %4

4:                                                ; preds = %1, %0
  ret void
}

