class RVInst_DUALADD<dag outs, dag ins> : RVInst<outs, ins, "dualadd", "$rd1, $rd2, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd2;
	bits<5> rd1;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = rs1{4-0};
	let Inst{19-15} = rd2{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd1{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LOAD_ACC<dag outs, dag ins> : RVInst<outs, ins, "load.acc", "$rd1, $rd2, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd2;
	bits<5> rd1;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = rs1{4-0};
	let Inst{19-15} = rd2{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd1{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_DOTP<dag outs, dag ins> : RVInst<outs, ins, "dotp", "$rd, $rs1, $rs2", [], InstFormatOther> {
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
class RVInst_UNPACK16<dag outs, dag ins> : RVInst<outs, ins, "unpack16", "$rd1, $rd2, $rs1", [], InstFormatOther> {
	bits<5> rs1;
	bits<5> rd2;
	bits<5> rd1;
	let Inst{31-25} = 0x0;
	let Inst{24-20} = rs1{4-0};
	let Inst{19-15} = rd2{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd1{4-0};
	let Inst{6-0} = 0x0;
}
