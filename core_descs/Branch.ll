; ModuleID = 'mod'
source_filename = "mod"

define void @implBMASK(ptr %rs1, ptr %rs2, i32 %imm12) {
  %1 = and i32 %imm12, 4095
  %2 = icmp eq i32 %imm12, %1
  call void @llvm.assume(i1 %2)
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %3 = and i32 %rs1.v, %rs2.v
  %4 = call i32 @llvm.riscv.pg.branch.i32(i32 %3)
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #0

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare i32 @llvm.riscv.pg.branch.i32(i32) #1

attributes #0 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #1 = { nocallback nofree nosync nounwind willreturn }

