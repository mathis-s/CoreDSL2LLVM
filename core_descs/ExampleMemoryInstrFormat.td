class RVInst_LDR<dag outs, dag ins> : RVInst<outs, ins, "ldr", "$rd, $imm $rs2($rs1)", [], InstFormatOther> {
	bits<5> imm;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = imm{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_STR<dag outs, dag ins> : RVInst<outs, ins, "str", "$rs3, $imm $rs2($rs1)", [], InstFormatOther> {
	bits<5> rs3;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> imm;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = rs3{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = imm{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_ADD3<dag outs, dag ins> : RVInst<outs, ins, "add3", "$rd, $rs2($rs1)", [], InstFormatOther> {
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
class RVInst_LOADMAC<dag outs, dag ins> : RVInst<outs, ins, "loadmac", "$rd, ($rs2), ($rs1)", [], InstFormatOther> {
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
