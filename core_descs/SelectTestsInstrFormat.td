class RVInst_SEXT_BIT23<dag outs, dag ins> : RVInst<outs, ins, "sext.bit23", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
class RVInst_ZEXT_BIT23<dag outs, dag ins> : RVInst<outs, ins, "zext.bit23", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x1;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
class RVInst_ZEXT_UNALIGNED_BYTE<dag outs, dag ins> : RVInst<outs, ins, "zext.unaligned.byte", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x2;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
class RVInst_SEXT_UNALIGNED_BYTE<dag outs, dag ins> : RVInst<outs, ins, "sext.unaligned.byte", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x3;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
class RVInst_ZEXT_ALIGNED_WORD<dag outs, dag ins> : RVInst<outs, ins, "zext.aligned.word", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x4;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
class RVInst_SEXT_ALIGNED_WORD<dag outs, dag ins> : RVInst<outs, ins, "sext.aligned.word", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-27} = 0x0;
	let Inst{26-26} = 0x0;
	let Inst{25-25} = 0x0;
	let Inst{24-20} = 0x5;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x7b;
}
