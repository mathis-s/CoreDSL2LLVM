class RVInst_ADDW2<dag outs, dag ins> : RVInst<outs, ins, "addw2", "$rd, $rs1, $rs2", [], InstFormatOther> {
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
class RVInst_ADDW3<dag outs, dag ins> : RVInst<outs, ins, "addw3", "$rd, $rs1, $rs2", [], InstFormatOther> {
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
class RVInst_LB<dag outs, dag ins> : RVInst<outs, ins, "lb", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LH<dag outs, dag ins> : RVInst<outs, ins, "lh", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LW<dag outs, dag ins> : RVInst<outs, ins, "lw", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LD<dag outs, dag ins> : RVInst<outs, ins, "ld", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LBU<dag outs, dag ins> : RVInst<outs, ins, "lbu", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LHU<dag outs, dag ins> : RVInst<outs, ins, "lhu", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
class RVInst_LWU<dag outs, dag ins> : RVInst<outs, ins, "lwu", "$rd, {imm}($rs1)", [], InstFormatOther> {
	bits<12> imm;
	bits<5> rs1;
	bits<5> rd;
	let Inst{31-20} = imm{11-0};
	let Inst{19-15} = rs1{4-0};
	let Inst{14-12} = 0x0;
	let Inst{11-7} = rd{4-0};
	let Inst{6-0} = 0x0;
}
