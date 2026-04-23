#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int threshold = 0;

// Experiment 1 - threshold sweep (n=10^5): parallel time by threshold
//   threshold=10:     0.082 s  (slower than serial; too many tasks)
//   threshold=100:    0.026 s
//   threshold=500:    0.025 s
//   threshold=5000:   0.025 s  <- optimal
//   threshold=50000:  0.037 s
//   threshold=100000: 0.057 s  (~serial)
//   serial:           0.050 s
//
// Experiment 2 - n sweep (threshold=500): crossover point
//   n=100:       serial 0.000032 s, parallel 0.001871 s  (parallel 58x slower)
//   n=1000:      serial 0.000327 s, parallel 0.001903 s  (parallel  6x slower)
//   n=10000:     serial 0.003772 s, parallel 0.003487 s  (~equal, crossover)
//   n=100000:    serial 0.044398 s, parallel 0.021134 s  (parallel  2x faster)
//   n=1000000:   serial 0.506734 s, parallel 0.072602 s  (parallel  7x faster)
//
// Conclusion: spawning a task per recursive call is too costly for small subproblems.
// threshold=10 is even slower than serial; optimal threshold (~5000 for n=10^5)
// balances parallelism and task overhead. Parallel beats serial for n >= ~10^4.
//
// Note: These experiments are conducted in code server (interactive) on TSUBAME 4.0.
//       The results may vary on different machines.

void merge(std::vector<int>& vec, int begin, int mid, int end) {
  std::vector<int> tmp(end-begin+1);
  int left = begin;
  int right = mid+1;
  for (int i=0; i<tmp.size(); i++) {
    if (left > mid)
      tmp[i] = vec[right++];
    else if (right > end)
      tmp[i] = vec[left++];
    else if (vec[left] <= vec[right])
      tmp[i] = vec[left++];
    else
      tmp[i] = vec[right++];
  }
  for (int i=0; i<tmp.size(); i++)
    vec[begin++] = tmp[i];
}

void merge_sort(std::vector<int>& vec, int begin, int end) {
  if(begin < end) {
    int mid = (begin + end) / 2;

    if (end - begin > threshold) {
      #pragma omp task shared(vec)
      merge_sort(vec, begin, mid);

      #pragma omp task shared(vec)
      merge_sort(vec, mid+1, end);

      #pragma omp taskwait
    } else {
      merge_sort(vec, begin, mid);
      merge_sort(vec, mid+1, end);
    }

    merge(vec, begin, mid, end);
  }
}

int main(int argc, char **argv) {
  int parallel = (argc > 1) ? atoi(argv[1]) : 1;

  int n = 20;
  std::vector<int> vec(n);
  for (int i=0; i<n; i++) {
    vec[i] = rand() % (10 * n);
    printf("%d ",vec[i]);
  }
  printf("\n");

  double t0 = omp_get_wtime();
  #pragma omp parallel if(parallel)
  {
    #pragma omp single
    merge_sort(vec, 0, n-1);
  }
  double t1 = omp_get_wtime();
  for (int i=0; i<n; i++) {
    printf("%d ",vec[i]);
  }
  printf("\n");
  printf("time: %f s (%s)\n", t1 - t0, parallel ? "parallel" : "serial");
}
