class RVInst_CV_SUBINCACC<dag outs, dag ins> : RVInst<outs, ins, "cv.subincacc", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x28;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = rd{4-0};
	let Opcode = 0x2b;
}
