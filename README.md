# CoreDSLToLLVM for GlobalISel
This repo implements an LLVM-based tool for automatic generation of Instruction Selection Patterns for GlobalISel.
Patterns are generated from CoreDSL2 descriptions of instruction behavior.

## Build Process
Everything is tied into the [LLVM build system](https://llvm.org/docs/CMake.html). Generally:

1. Clone the repo and `cd` into it.
2. `mkdir llvm/build && cd llvm/build`
3. `cmake ..`
4. `cmake --build .`

## Usage
Pattern Generation is implemented as an additional LLVM tool (`pattern-gen`) within the LLVM tree. Run `./llvm/build/bin/pattern-gen core_descs/Example.core_desc` to generate a pattern for an example instruction.

## Example


### CoreDSL2 Instruction Description
```verilog
CV_SUBINCACC {
  encoding: 7'b0101000 :: 5'b00000 :: rs1[4:0] :: 3'b011 :: rd[4:0] :: 7'b0101011;
  assembly: "{name(rd)}, {name(rs1)}";
  behavior: {
    if (rd != 0) {
      X[rd] += X[rs1] - X[rs2] + 1;
    }
  }
}
```

### LLVM-IR
```llvm
define void @implCV_SUBINCACC(ptr noalias nocapture %rd, ptr nocapture readonly %rs1, ptr nocapture readonly %rs2, i32 %imm, i32 %imm2) local_unnamed_addr #0 {
  %rs1.v = load i32, ptr %rs1, align 4
  %rs2.v = load i32, ptr %rs2, align 4
  %rd.v = load i32, ptr %rd, align 4
  %1 = add i32 %rs1.v, 1
  %2 = sub i32 %1, %rs2.v
  %3 = add i32 %2, %rd.v
  store i32 %3, ptr %rd, align 4
  ret void
}
```

### pre-ISel MIR
```
bb.1 (%ir-block.0):
  liveins: $x10, $x11, $x12, $x13, $x14
  %0:gprb(p0) = COPY $x10
  %1:gprb(p0) = COPY $x11
  %2:gprb(p0) = COPY $x12
  %3:gprb(s32) = COPY $x13
  %4:gprb(s32) = COPY $x14
  %8:gprb(s32) = G_CONSTANT i32 1
  %5:gprb(s32) = G_LOAD %1:gprb(p0) :: (load (s32) from %ir.rs1)
  %6:gprb(s32) = G_LOAD %2:gprb(p0) :: (load (s32) from %ir.rs2)
  %7:gprb(s32) = G_LOAD %0:gprb(p0) :: (load (s32) from %ir.rd)
  %9:gprb(s32) = G_ADD %5:gprb, %8:gprb
  %10:gprb(s32) = G_SUB %9:gprb, %6:gprb
  %11:gprb(s32) = G_ADD %10:gprb, %7:gprb
  G_STORE %11:gprb(s32), %0:gprb(p0) :: (store (s32) into %ir.rd)
  PseudoRET
```

### Generated Pattern
```llvm
def : Pat<
  (add (sub (add GPR:$rs1, (i32 1)), GPR:$rs2), GPR:$rd),
  (CV_SUBINCACC_S_S_S_S GPR:$rd, GPR:$rs1, GPR:$rs2)>;
```
