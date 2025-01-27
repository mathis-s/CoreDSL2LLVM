// ./llvm/build/bin/clang -S -O3 -target riscv32-unknown-elf -march=rv32imac_xcvalu test_programs/dual.c -mllvm -global-isel -mllvm --global-isel-abort=1

typedef struct {int a; int b;} Pair;

Pair dual_add(int a, int b, int c)
{
    return (Pair){a + c, b + c};
}