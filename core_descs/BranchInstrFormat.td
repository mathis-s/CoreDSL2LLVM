class RVInst_BMASK<dag outs, dag ins> : RVInst<outs, ins, "bmask", "$rs1, $rs2, $imm12", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rs2;
	bits<12> imm12;
	let Inst{31-25} = imm12{6-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x3;
	let Inst{11-7} = imm12{11-7};
	let Inst{6-0} = 0x2b;
}
