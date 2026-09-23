// SPDX-License-Identifier: 0BSD

package infer

// dotScalar returns the inner product of a and b.
func dotScalar(a, b []float32) float32 {
	var s0, s1, s2, s3 float32
	n := len(a) &^ 3
	for i := 0; i < n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	for i := n; i < len(a); i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

// dot4Scalar computes the inner product of w with four vectors at once.
func dot4Scalar(w, a, b, c, d []float32) (s0, s1, s2, s3 float32) {
	for i := range w {
		v := w[i]
		s0 += v * a[i]
		s1 += v * b[i]
		s2 += v * c[i]
		s3 += v * d[i]
	}
	return
}
