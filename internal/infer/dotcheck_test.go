package infer

import (
	"math/rand"
	"testing"
)

func TestDotAsm(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	r := rand.New(rand.NewSource(1))
	for _, n := range []int{8, 16, 40, 128, 1024, 1027} {
		a := make([]float32, n)
		b := make([]float32, n)
		c := make([]float32, n)
		d := make([]float32, n)
		w := make([]float32, n)
		for i := range a {
			a[i], b[i], c[i], d[i], w[i] = r.Float32(), r.Float32(), r.Float32(), r.Float32(), r.Float32()
		}
		got := dot(a, b)
		want := dotScalar(a, b)
		if abs(got-want) > 1e-3*abs(want)+1e-6 {
			t.Fatalf("dot n=%d: got %v want %v", n, got, want)
		}
		g0, g1, g2, g3 := dot4(w, a, b, c, d)
		w0, w1, w2, w3 := dot4Scalar(w, a, b, c, d)
		if abs(g0-w0) > 1e-3*abs(w0)+1e-6 || abs(g1-w1) > 1e-3*abs(w1)+1e-6 {
			t.Fatalf("dot4 n=%d: got %v want %v", n, []float32{g0, g1, g2, g3}, []float32{w0, w1, w2, w3})
		}
	}
}

func abs(x float32) float32 {
	if x < 0 {
		return -x
	}
	return x
}
