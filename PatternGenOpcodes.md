# Generic Opcodes

Helpful list from: `lvm-project/llvm/include/llvm/Target/GlobalISel/SelectionDAGCompat.td`

Warning: dropped special conditions and preds!

handled generic instructions are marked with a `>`
TODOs are marked with a `!`
Maybes are marked with a `?`
FPs are marked with a `%`
Ignores are marked with a `#`

Missing in map: `G_SHUFFLE_VECTOR (?)`

```
>def : GINodeEquiv<G_ANYEXT, anyext>;
>def : GINodeEquiv<G_SEXT, sext>;
>def : GINodeEquiv<G_ZEXT, zext>;
>def : GINodeEquiv<G_TRUNC, trunc>;
>def : GINodeEquiv<G_BITCAST, bitconvert>;
// G_INTTOPTR - SelectionDAG has no equivalent.
// G_PTRTOINT - SelectionDAG has no equivalent.
> def : GINodeEquiv<G_CONSTANT, imm>;
// timm must not be materialized and therefore has no GlobalISel equivalent
%def : GINodeEquiv<G_FCONSTANT, fpimm>;
>def : GINodeEquiv<G_IMPLICIT_DEF, undef>;
#def : GINodeEquiv<G_FRAME_INDEX, frameindex>;
#def : GINodeEquiv<G_BLOCK_ADDR, blockaddress>;
>def : GINodeEquiv<G_PTR_ADD, ptradd>;
>def : GINodeEquiv<G_ADD, add>;
>def : GINodeEquiv<G_SUB, sub>;
>def : GINodeEquiv<G_MUL, mul>;
>def : GINodeEquiv<G_UMULH, mulhu>;
>def : GINodeEquiv<G_SMULH, mulhs>;
>def : GINodeEquiv<G_SDIV, sdiv>;
>def : GINodeEquiv<G_UDIV, udiv>;
>def : GINodeEquiv<G_SREM, srem>;
>def : GINodeEquiv<G_UREM, urem>;
>def : GINodeEquiv<G_AND, and>;
>def : GINodeEquiv<G_OR, or>;
>def : GINodeEquiv<G_XOR, xor>;
>def : GINodeEquiv<G_SHL, shl>;
>def : GINodeEquiv<G_LSHR, srl>;
>def : GINodeEquiv<G_ASHR, sra>;
>def : GINodeEquiv<G_SADDSAT, saddsat>;
>def : GINodeEquiv<G_UADDSAT, uaddsat>;
>def : GINodeEquiv<G_SSUBSAT, ssubsat>;
>def : GINodeEquiv<G_USUBSAT, usubsat>;
>def : GINodeEquiv<G_SSHLSAT, sshlsat>;
>def : GINodeEquiv<G_USHLSAT, ushlsat>;
>def : GINodeEquiv<G_SMULFIX, smulfix>;
>def : GINodeEquiv<G_UMULFIX, umulfix>;
>def : GINodeEquiv<G_SMULFIXSAT, smulfixsat>;
>def : GINodeEquiv<G_UMULFIXSAT, umulfixsat>;
>def : GINodeEquiv<G_SDIVFIX, sdivfix>;
>def : GINodeEquiv<G_UDIVFIX, udivfix>;
>def : GINodeEquiv<G_SDIVFIXSAT, sdivfixsat>;
>def : GINodeEquiv<G_UDIVFIXSAT, udivfixsat>;
>def : GINodeEquiv<G_SELECT, select>;
?def : GINodeEquiv<G_SELECT, vselect>;
%def : GINodeEquiv<G_FNEG, fneg>;
%def : GINodeEquiv<G_FPEXT, fpextend>;
%def : GINodeEquiv<G_FPTRUNC, fpround>;
%def : GINodeEquiv<G_FPTOSI, fp_to_sint>;
%def : GINodeEquiv<G_FPTOUI, fp_to_uint>;
%def : GINodeEquiv<G_SITOFP, sint_to_fp>;
%def : GINodeEquiv<G_UITOFP, uint_to_fp>;
%def : GINodeEquiv<G_FADD, fadd>;
%def : GINodeEquiv<G_FSUB, fsub>;
%def : GINodeEquiv<G_FMA, fma>;
%def : GINodeEquiv<G_FMAD, fmad>;
%def : GINodeEquiv<G_FMUL, fmul>;
%def : GINodeEquiv<G_FDIV, fdiv>;
%def : GINodeEquiv<G_FREM, frem>;
%def : GINodeEquiv<G_FPOW, fpow>;
%def : GINodeEquiv<G_FEXP2, fexp2>;
%def : GINodeEquiv<G_FEXP10, fexp10>;
%def : GINodeEquiv<G_FLOG2, flog2>;
%def : GINodeEquiv<G_FLDEXP, fldexp>;
%def : GINodeEquiv<G_FCANONICALIZE, fcanonicalize>;
%def : GINodeEquiv<G_IS_FPCLASS, is_fpclass>;
#def : GINodeEquiv<G_INTRINSIC, intrinsic_wo_chain>;
%def : GINodeEquiv<G_GET_FPENV, get_fpenv>;
%def : GINodeEquiv<G_SET_FPENV, set_fpenv>;
%def : GINodeEquiv<G_RESET_FPENV, reset_fpenv>;
%def : GINodeEquiv<G_GET_FPMODE, get_fpmode>;
%def : GINodeEquiv<G_SET_FPMODE, set_fpmode>;
%def : GINodeEquiv<G_RESET_FPMODE, reset_fpmode>;
#def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_void>;
#def : GINodeEquiv<G_INTRINSIC_W_SIDE_EFFECTS, intrinsic_w_chain>;
?def : GINodeEquiv<G_BR, br>;
?def : GINodeEquiv<G_BRCOND, brcond>;
>def : GINodeEquiv<G_BSWAP, bswap>;
>def : GINodeEquiv<G_BITREVERSE, bitreverse>;
>def : GINodeEquiv<G_FSHL, fshl>;
>def : GINodeEquiv<G_FSHR, fshr>;
>def : GINodeEquiv<G_CTLZ, ctlz>;
>def : GINodeEquiv<G_CTTZ, cttz>;
>def : GINodeEquiv<G_CTLZ_ZERO_UNDEF, ctlz_zero_undef>;
>def : GINodeEquiv<G_CTTZ_ZERO_UNDEF, cttz_zero_undef>;
>def : GINodeEquiv<G_CTPOP, ctpop>;
>def : GINodeEquiv<G_EXTRACT_VECTOR_ELT, extractelt>;
>def : GINodeEquiv<G_INSERT_VECTOR_ELT, vector_insert>;
?def : GINodeEquiv<G_CONCAT_VECTORS, concat_vectors>;
>def : GINodeEquiv<G_BUILD_VECTOR, build_vector>;
%def : GINodeEquiv<G_FCEIL, fceil>;
%def : GINodeEquiv<G_FCOS, fcos>;
%def : GINodeEquiv<G_FSIN, fsin>;
%def : GINodeEquiv<G_FTAN, ftan>;
%def : GINodeEquiv<G_FACOS, facos>;
%def : GINodeEquiv<G_FASIN, fasin>;
%def : GINodeEquiv<G_FATAN, fatan>;
%def : GINodeEquiv<G_FCOSH, fcosh>;
%def : GINodeEquiv<G_FSINH, fsinh>;
%def : GINodeEquiv<G_FTANH, ftanh>;
%def : GINodeEquiv<G_FABS, fabs>;
%def : GINodeEquiv<G_FSQRT, fsqrt>;
%def : GINodeEquiv<G_FFLOOR, ffloor>;
%def : GINodeEquiv<G_FRINT, frint>;
%def : GINodeEquiv<G_FNEARBYINT, fnearbyint>;
%def : GINodeEquiv<G_INTRINSIC_TRUNC, ftrunc>;
%def : GINodeEquiv<G_INTRINSIC_ROUND, fround>;
%def : GINodeEquiv<G_INTRINSIC_ROUNDEVEN, froundeven>;
#def : GINodeEquiv<G_INTRINSIC_LRINT, lrint>;
#def : GINodeEquiv<G_INTRINSIC_LLRINT, llrint>;
%def : GINodeEquiv<G_FCOPYSIGN, fcopysign>;
>def : GINodeEquiv<G_SMIN, smin>;
>def : GINodeEquiv<G_SMAX, smax>;
>def : GINodeEquiv<G_UMIN, umin>;
>def : GINodeEquiv<G_UMAX, umax>;
>def : GINodeEquiv<G_ABS, abs>;
%def : GINodeEquiv<G_FMINNUM, fminnum>;
%def : GINodeEquiv<G_FMAXNUM, fmaxnum>;
%def : GINodeEquiv<G_FMINNUM_IEEE, fminnum_ieee>;
%def : GINodeEquiv<G_FMAXNUM_IEEE, fmaxnum_ieee>;
%def : GINodeEquiv<G_FMAXIMUM, fmaximum>;
%def : GINodeEquiv<G_FMINIMUM, fminimum>;
#def : GINodeEquiv<G_READCYCLECOUNTER, readcyclecounter>;
#def : GINodeEquiv<G_READSTEADYCOUNTER, readsteadycounter>;
>def : GINodeEquiv<G_ROTR, rotr>;
>def : GINodeEquiv<G_ROTL, rotl>;
?def : GINodeEquiv<G_LROUND, lround>;
?def : GINodeEquiv<G_LLROUND, llround>;
%def : GINodeEquiv<G_VECREDUCE_FADD, vecreduce_fadd>;
%def : GINodeEquiv<G_VECREDUCE_FMAX, vecreduce_fmax>;
%def : GINodeEquiv<G_VECREDUCE_FMIN, vecreduce_fmin>;
%def : GINodeEquiv<G_VECREDUCE_FMAXIMUM, vecreduce_fmaximum>;
%def : GINodeEquiv<G_VECREDUCE_FMINIMUM, vecreduce_fminimum>;
?def : GINodeEquiv<G_VECREDUCE_UMIN, vecreduce_umin>;
?def : GINodeEquiv<G_VECREDUCE_UMAX, vecreduce_umax>;
?def : GINodeEquiv<G_VECREDUCE_SMIN, vecreduce_smin>;
?def : GINodeEquiv<G_VECREDUCE_SMAX, vecreduce_smax>;
>def : GINodeEquiv<G_VECREDUCE_ADD, vecreduce_add>;
?def : GINodeEquiv<G_VECTOR_COMPRESS, vector_compress>;
%def : GINodeEquiv<G_STRICT_FADD, strict_fadd>;
%def : GINodeEquiv<G_STRICT_FSUB, strict_fsub>;
%def : GINodeEquiv<G_STRICT_FMUL, strict_fmul>;
%def : GINodeEquiv<G_STRICT_FDIV, strict_fdiv>;
%def : GINodeEquiv<G_STRICT_FREM, strict_frem>;
%def : GINodeEquiv<G_STRICT_FMA, strict_fma>;
%def : GINodeEquiv<G_STRICT_FSQRT, strict_fsqrt>;
%def : GINodeEquiv<G_STRICT_FLDEXP, strict_fldexp>;
>def : GINodeEquiv<G_LOAD, ld>;
>def : GINodeEquiv<G_ICMP, setcc>;
>def : GINodeEquiv<G_STORE, st>;
#def : GINodeEquiv<G_STORE, atomic_store>;
#def : GINodeEquiv<G_LOAD, atomic_load>;
#def : GINodeEquiv<G_ATOMIC_CMPXCHG, atomic_cmp_swap>;
#def : GINodeEquiv<G_ATOMICRMW_XCHG, atomic_swap>;
#def : GINodeEquiv<G_ATOMICRMW_ADD, atomic_load_add>;
#def : GINodeEquiv<G_ATOMICRMW_SUB, atomic_load_sub>;
#def : GINodeEquiv<G_ATOMICRMW_AND, atomic_load_and>;
#def : GINodeEquiv<G_ATOMICRMW_NAND, atomic_load_nand>;
#def : GINodeEquiv<G_ATOMICRMW_OR, atomic_load_or>;
#def : GINodeEquiv<G_ATOMICRMW_XOR, atomic_load_xor>;
#def : GINodeEquiv<G_ATOMICRMW_MIN, atomic_load_min>;
#def : GINodeEquiv<G_ATOMICRMW_MAX, atomic_load_max>;
#def : GINodeEquiv<G_ATOMICRMW_UMIN, atomic_load_umin>;
#def : GINodeEquiv<G_ATOMICRMW_UMAX, atomic_load_umax>;
#def : GINodeEquiv<G_ATOMICRMW_FADD, atomic_load_fadd>;
#def : GINodeEquiv<G_ATOMICRMW_FSUB, atomic_load_fsub>;
#def : GINodeEquiv<G_ATOMICRMW_FMAX, atomic_load_fmax>;
#def : GINodeEquiv<G_ATOMICRMW_FMIN, atomic_load_fmin>;
#def : GINodeEquiv<G_ATOMICRMW_UINC_WRAP, atomic_load_uinc_wrap>;
#def : GINodeEquiv<G_ATOMICRMW_UDEC_WRAP, atomic_load_udec_wrap>;
#def : GINodeEquiv<G_FENCE, atomic_fence>;
#def : GINodeEquiv<G_PREFETCH, prefetch>;
#def : GINodeEquiv<G_TRAP, trap>;
#def : GINodeEquiv<G_DEBUGTRAP, debugtrap>;
#def : GINodeEquiv<G_UBSANTRAP, ubsantrap>;
```
