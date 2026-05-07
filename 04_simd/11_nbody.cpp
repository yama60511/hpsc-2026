#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  alignas(64) float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  __m512 one = _mm512_set1_ps(1);
  __m512 zero = _mm512_setzero_ps();

  for(int i=0; i<N; i++) {
// [Legacy] Scalar inner loop over j
//    for(int j=0; j<N; j++) {
//      if(i != j) {
//        float rx = x[i] - x[j];
//        float ry = y[i] - y[j];
//        float r = std::sqrt(rx * rx + ry * ry);
//        fx[i] -= rx * m[j] / (r * r * r);
//        fy[i] -= ry * m[j] / (r * r * r);
//      }
//    }

    // ===== Change: Vectorize the j loop with one 512-bit register =====
    __mmask16 not_self = static_cast<__mmask16>(0xffffu & ~(1u << i));
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 rxvec = _mm512_sub_ps(xivec, xvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yvec);
    __m512 r2vec = _mm512_add_ps(_mm512_mul_ps(rxvec, rxvec),
                                 _mm512_mul_ps(ryvec, ryvec));
    __m512 safe_r2vec = _mm512_mask_blend_ps(not_self, one, r2vec);
    __m512 inv_rvec = _mm512_rsqrt14_ps(safe_r2vec);
    __m512 inv_r2vec = _mm512_mul_ps(inv_rvec, inv_rvec);
    __m512 inv_r3vec = _mm512_mul_ps(inv_r2vec, inv_rvec);
    __m512 scalevec = _mm512_mul_ps(mvec, inv_r3vec);
    __m512 fxvec = _mm512_mul_ps(rxvec, scalevec);
    __m512 fyvec = _mm512_mul_ps(ryvec, scalevec);
    fxvec = _mm512_mask_blend_ps(not_self, zero, fxvec);
    fyvec = _mm512_mask_blend_ps(not_self, zero, fyvec);
    fx[i] -= _mm512_reduce_add_ps(fxvec);
    fy[i] -= _mm512_reduce_add_ps(fyvec);
    // =================================================================

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
