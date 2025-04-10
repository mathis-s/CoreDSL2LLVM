

extern int something();

void do_branch() {
  int a = -1;
  int b = 1000;

  do {
    a = something();
    b--;
  }
  while (a & b);
}
