class RVInst_CV_MAC<dag outs, dag ins> : RVInst<outs, ins, "cv.mac", "$rd, $rs1, $rs2", [], InstFormatOther> {
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
class RVInst_CV_ABS<dag outs, dag ins> : RVInst<outs, ins, "cv.abs", "$rd, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = 0x0;
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_CV_ADDN<dag outs, dag ins> : RVInst<outs, ins, "cv.addn", "$rd, $rs1, $rs2, {Luimm5}", [], InstFormatOther> {
	bits<5> Luimm5;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = Luimm5{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
