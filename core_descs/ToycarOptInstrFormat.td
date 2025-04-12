class RVInst_MAC_LD_X2_n2<dag outs, dag ins> : RVInst<outs, ins, "mac.ld.x2", "$rd, $rd2, $rs1, -2($rs2)", [], InstFormatOther> {
	bits<5> rd2;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x2;
	let Inst{29-25} = rd2{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x2b;
}
class RVInst_MAC_LD_X2_n1<dag outs, dag ins> : RVInst<outs, ins, "mac.ld.x2", "$rd, $rd2, $rs1, -1($rs2)", [], InstFormatOther> {
	bits<5> rd2;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x3;
	let Inst{29-25} = rd2{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x2b;
}
class RVInst_MAC_LD_X2_0<dag outs, dag ins> : RVInst<outs, ins, "mac.ld.x2", "$rd, $rd2, $rs1, 0($rs2)", [], InstFormatOther> {
	bits<5> rd2;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x0;
	let Inst{29-25} = rd2{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x2b;
}
class RVInst_MAC_LD_X2_1<dag outs, dag ins> : RVInst<outs, ins, "mac.ld.x2", "$rd, $rd2, $rs1, 1($rs2)", [], InstFormatOther> {
	bits<5> rd2;
	bits<5> rs2;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-30} = 0x1;
	let Inst{29-25} = rd2{4-0};
	let Inst{24-20} = rs2{4-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x2;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x2b;
}
