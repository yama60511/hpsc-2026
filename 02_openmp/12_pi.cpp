#include <cstdio>
#include <cstdlib>
#include <omp.h>

int main(int argc, char **argv) {
  // Serial outperforms parallel for n < 10^5 due to thread overhead (tested at n = 10^1, 10^2, ...).
  int n = 10;
  int parallel = (argc > 1) ? atoi(argv[1]) : 1;
  double dx = 1. / n;
  double pi = 0;
  double t0 = omp_get_wtime();
#pragma omp parallel for reduction(+:pi) if(parallel)
  for (int i=0; i<n; i++) {
    double x = (i + 0.5) * dx;
    pi += 4.0 / (1.0 + x * x) * dx;
  }
  double t1 = omp_get_wtime();
  printf("%17.15f\n",pi);
  printf("time: %f s (%s)\n", t1 - t0, parallel ? "parallel" : "serial");
}
