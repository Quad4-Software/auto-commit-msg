// SPDX-License-Identifier: 0BSD

//go:build !amd64

package infer

import "quad4.io/auto-commit-msg/internal/gguf"

const hasAVX2 = false

func dequantRowFast(t *gguf.Tensor, r int, dst []float32) {
	dequantRowSlow(t, r, dst)
}

func dot(a, b []float32) float32 {
	return dotScalar(a, b)
}

func dot4(w, a, b, c, d []float32) (s0, s1, s2, s3 float32) {
	return dot4Scalar(w, a, b, c, d)
}

func axpy(dst []float32, w float32, src []float32) {
	for i, v := range src {
		dst[i] += w * v
	}
}
