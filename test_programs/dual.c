// ./llvm/build/bin/clang -S -O3 -target riscv32-unknown-elf -march=rv32imac_xcvalu test_programs/dual.c -mllvm -global-isel -mllvm --global-isel-abort=1
#include <stddef.h>
typedef struct {int a; int b;} Pair;

Pair dual_add(int a, int b, int c)
{
    return (Pair){a + c, b + c};
}

int acc(size_t n, int arr[n])
{
    int acc = 0;
    for (size_t i = 0; i < n; i++)
        acc += arr[i];
    return acc;
}
