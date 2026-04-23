#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main(int argc, char **argv) {
  int parallel = (argc > 1) ? atoi(argv[1]) : 1;
  // Create a random array
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  // Put the array into buckets
  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  // Generate bucket offsets (exclusive prefix sum)
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
  // Reconstruct the sorted array
  double t0 = omp_get_wtime();
  if (!parallel) {
    for (int i=0; i<range; i++) {
      int j = offset[i];
      for (; bucket[i]>0; bucket[i]--) {
        key[j++] = i;
      }
    }
  } else {
// Parallel version 3
#pragma omp parallel
      for (int r = 0; r < range; r++) {
        int start = offset[r];
        int end = (r == range - 1) ? n : offset[r + 1];
#pragma omp for nowait
        for (int p = start; p < end; p++) {
            key[p] = r;
        }
      }
// Parallel version 1
//    std::vector<int> mark(n,0);
//    for (int i=1; i<range; i++)
//      mark[offset[i]] = 1;
//
//    std::vector<int> b(n);
//#pragma omp parallel
//    for (int j=1; j<n; j<<=1) {
//#pragma omp for
//      for (int i=0; i<n; i++)
//        b[i] = mark[i];
//#pragma omp for
//      for (int i=j; i<n; i++)
//        mark[i] += b[i-j];
//    }
//#pragma omp parallel for
//    for (int i=0; i<n; i++)
//      key[i] = mark[i];

// Parallel version 2
//#pragma omp parallel for
//    for (int p = 0; p < n; p++) {
//      int r = 0;
//      while (r < range - 1 && p >= offset[r + 1]) {
//        r++;
//      }
//      key[p] = r;
//    }
  }
  double t1 = omp_get_wtime();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
  printf("time: %f s (%s)\n", t1 - t0, parallel ? "parallel" : "serial");
}
