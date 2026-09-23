// SPDX-License-Identifier: 0BSD

//go:build amd64

package infer

func cpuHasAVX2() bool

// dotAVX2 computes sum(a[i]*b[i]) for i < n, n % 8 == 0.
func dotAVX2(a, b *float32, n int) float32

// dot4AVX2 computes four dots of w against a, b, c, d for i < n, n % 8 == 0.
func dot4AVX2(w, a, b, c, d *float32, n int) (s0, s1, s2, s3 float32)

// axpyAVX2 does dst[i] += w * src[i] for i < n, n % 8 == 0.
func axpyAVX2(dst *float32, w float32, src *float32, n int)

var hasAVX2 = cpuHasAVX2()

// axpy accumulates dst[i] += w * src[i].
func axpy(dst []float32, w float32, src []float32) {
	n := len(src) &^ 7
	if hasAVX2 && n > 0 {
		axpyAVX2(&dst[0], w, &src[0], n)
	} else {
		n = 0
	}
	for ; n < len(src); n++ {
		dst[n] += w * src[n]
	}
}

func dot(a, b []float32) float32 {
	if !hasAVX2 {
		return dotScalar(a, b)
	}
	n := len(a) &^ 7
	var s float32
	if n > 0 {
		s = dotAVX2(&a[0], &b[0], n)
	}
	for ; n < len(a); n++ {
		s += a[n] * b[n]
	}
	return s
}

func dot4(w, a, b, c, d []float32) (s0, s1, s2, s3 float32) {
	if !hasAVX2 {
		return dot4Scalar(w, a, b, c, d)
	}
	n := len(w) &^ 7
	if n > 0 {
		s0, s1, s2, s3 = dot4AVX2(&w[0], &a[0], &b[0], &c[0], &d[0], n)
	}
	for ; n < len(w); n++ {
		v := w[n]
		s0 += v * a[n]
		s1 += v * b[n]
		s2 += v * c[n]
		s3 += v * d[n]
	}
	return
}
