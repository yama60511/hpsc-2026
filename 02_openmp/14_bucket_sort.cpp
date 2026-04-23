#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
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
//  for (int i=0; i<range; i++) {
//    int j = offset[i];
//    for (; bucket[i]>0; bucket[i]--) {
//      key[j++] = i;
//    }
//  }
  std::vector<int> mark(n,0);
  for (int i=1; i<range; i++)
    mark[offset[i]] = 1;

  std::vector<int> b(n);
#pragma omp parallel
  for (int j=1; j<n; j<<=1) {
#pragma omp for
    for (int i=0; i<n; i++)
      b[i] = mark[i];
#pragma omp for
    for (int i=j; i<n; i++)
      mark[i] += b[i-j];
  }
#pragma omp parallel for
  for (int i=0; i<n; i++)
    key[i] = mark[i];

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
