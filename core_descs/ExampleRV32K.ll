; ModuleID = 'mod'
source_filename = "mod"

define void @implK_LLI(i32 %imm, ptr noalias %rd) {
  %1 = and i32 %imm, -1
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %4

3:                                                ; preds = %0
  store i32 %imm, ptr %rd, align 4
  br label %4

4:                                                ; preds = %3, %0
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

define void @implK_ADDI(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 16777215
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %5

3:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %4 = add i32 %rs1.v, %imm
  store i32 %4, ptr %rd, align 4
  br label %5

5:                                                ; preds = %3, %0
  ret void
}

define void @implK_ANDI(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 16777215
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %5

3:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %4 = and i32 %rs1.v, %imm
  store i32 %4, ptr %rd, align 4
  br label %5

5:                                                ; preds = %3, %0
  ret void
}

define void @implK_XORI(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 16777215
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %5

3:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %4 = xor i32 %rs1.v, %imm
  store i32 %4, ptr %rd, align 4
  br label %5

5:                                                ; preds = %3, %0
  ret void
}

define void @implK_ORI(i32 %imm, ptr %rs1, ptr noalias %rd) {
  %1 = and i32 %imm, 16777215
  %2 = icmp eq i32 %imm, %1
  call void @llvm.assume(i1 %2)
  br i1 true, label %3, label %5

3:                                                ; preds = %0
  %rs1.v = load i32, ptr %rs1, align 4
  %4 = or i32 %rs1.v, %imm
  store i32 %4, ptr %rd, align 4
  br label %5

5:                                                ; preds = %3, %0
  ret void
}

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }

