#include <cstdio>
#include <cstdlib>
#include <vector>

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

    #pragma omp task shared(vec)
    merge_sort(vec, begin, mid);

    #pragma omp task shared(vec)
    merge_sort(vec, mid+1, end);

    #pragma omp taskwait

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
  #pragma omp parallel{
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