class RVInst_CV_SUBINCACC<dag outs, dag ins> : RVInst<outs, ins, "cv.subincacc", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_CV_MAXU<dag outs, dag ins> : RVInst<outs, ins, "cv.maxu", "$rd, $rs1, $rs2", [], InstFormatOther> {
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
