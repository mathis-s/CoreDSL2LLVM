; ModuleID = 'mod'
source_filename = "mod"
target datalayout = "e-m:e-p:32:32-i64:64-n32-S128"
target triple = "riscv32-unknown-linux-gnu"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @implCV_SUBINCACC(ptr nocapture readonly %rs2, ptr nocapture readonly %rs1, ptr noalias nocapture %rd) local_unnamed_addr #0 {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %rd.v = load i32, ptr %rd, align 4
  %1 = add i32 %rs1.v, 1
  %narrow = sub i32 %1, %rs2.v
  %2 = add i32 %narrow, %rd.v
  store i32 %2, ptr %rd, align 4
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define void @implCV_MAXU(ptr nocapture readonly %rs2, ptr nocapture readonly %rs1, ptr noalias nocapture writeonly %rd) local_unnamed_addr #0 {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %1 = icmp ugt i32 %rs1.v, %rs2.v
  %2 = zext i1 %1 to i32
  store i32 %2, ptr %rd, align 4
  ret void
}

attributes #0 = { mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) }
