// SPDX-License-Identifier: 0BSD

package infer

import (
	"math"
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

func stat(name string, x []float32, t *testing.T) {
	var mn, mx, mean float32
	mn, mx = x[0], x[0]
	nan := 0
	for _, v := range x {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			nan++
			continue
		}
		mn = min(mn, v)
		mx = max(mx, v)
		mean += v
	}
	t.Logf("%s: min=%.4g max=%.4g mean=%.4g nan=%d", name, mn, mx, mean/float32(len(x)), nan)
}

func TestTrace(t *testing.T) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, err := gguf.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Load(f, 2048)
	if err != nil {
		t.Fatal(err)
	}
	c := &m.Cfg
	embd, nh, nkv, hd := c.NEmbd, c.NHead, c.HeadKV(), c.HeadDim
	kvd := nkv * hd
	qd := nh * hd

	// single token "Hello" = 9707
	x := make([]float32, embd)
	dequantRow(m.embd, 9707, x)
	stat("embed", x, t)

	h := make([]float32, embd)
	q := make([]float32, qd)
	k := make([]float32, kvd)
	v := make([]float32, kvd)
	att := make([]float32, qd)
	xb := make([]float32, embd)
	hb := make([]float32, c.NFF)
	hb2 := make([]float32, c.NFF)

	for _, l := range m.layers[:2] {
		rmsnorm(h, x, l.attnNW, c.RMSEps)
		stat("h", h, t)
		matvec(q, l.q, h)
		matvec(k, l.k, h)
		matvec(v, l.v, h)
		stat("q", q, t)
		stat("k", k, t)
		for head := 0; head < nh; head++ {
			hq := q[head*hd : head*hd+hd]
			if l.qNW != nil {
				rmsnorm(hq, hq, l.qNW, c.RMSEps)
			}
			rope(hq, 0, c.RopeDim, m.freqs)
		}
		for head := 0; head < nkv; head++ {
			hk := k[head*hd : head*hd+hd]
			if l.kNW != nil {
				rmsnorm(hk, hk, l.kNW, c.RMSEps)
			}
			rope(hk, 0, c.RopeDim, m.freqs)
		}
		kc := make([]float32, kvd)
		vc := make([]float32, kvd)
		copy(kc, k)
		copy(vc, v)
		// attention: single position, trivial
		scale := 1 / float32(math.Sqrt(float64(hd)))
		for head := 0; head < nh; head++ {
			qh := q[head*hd : head*hd+hd]
			kvh := head / (nh / nkv)
			kb := kvh * hd
			s := dot(qh, kc[kb:kb+hd]) * scale
			out := att[head*hd : head*hd+hd]
			w := float32(math.Exp(float64(s - s)))
			for d := 0; d < hd; d++ {
				out[d] = w * vc[kb+d]
			}
		}
		stat("att", att, t)
		matvec(xb, l.o, att)
		for i := range x {
			x[i] += xb[i]
		}
		stat("x+att", x, t)

		rmsnorm(h, x, l.ffnNW, c.RMSEps)
		matvec(hb, l.gate, h)
		matvec(hb2, l.up, h)
		for i, g := range hb {
			hb[i] = g * (1 / (1 + float32(math.Exp(-float64(g))))) * hb2[i]
		}
		matvec(xb, l.down, hb)
		for i := range x {
			x[i] += xb[i]
		}
		stat("x+ffn", x, t)
	}
	hf := make([]float32, embd)
	rmsnorm(hf, x, m.outNW, c.RMSEps)
	logits := make([]float32, c.Vocab)
	matvec(logits, m.out, hf)
	best := 0
	for i := range logits {
		if logits[i] > logits[best] {
			best = i
		}
	}
	t.Logf("top logit for 'Hello': id=%d %.2f", best, logits[best])
}
