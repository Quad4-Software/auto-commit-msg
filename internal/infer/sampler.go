// SPDX-License-Identifier: 0BSD

package infer

import (
	"fmt"
	"math"
	"math/rand/v2"
	"time"
)

type Sampler struct {
	Temp float64 // 0 = greedy
	TopK int
	TopP float64
	Rng  *rand.Rand
}

func greedy(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// Sample picks a token from logits.
func (s *Sampler) Sample(logits []float32) int {
	if s.Temp <= 0 {
		return greedy(logits)
	}
	type cand struct {
		id int
		p  float64
	}
	// keep at most 64 candidates; nucleus/top-k sampling never needs more
	n := 64
	if s.TopK > 0 && s.TopK < n {
		n = s.TopK
	}
	idx := topK(logits, n)
	cands := make([]cand, len(idx))
	var sum float64
	mx := float64(logits[idx[0]])
	for i, id := range idx {
		p := math.Exp((float64(logits[id]) - mx) / s.Temp)
		cands[i] = cand{id, p}
		sum += p
	}
	if s.TopP > 0 && s.TopP < 1 {
		acc := 0.0
		cut := n
		for i, c := range cands {
			acc += c.p / sum
			if acc >= s.TopP {
				cut = i + 1
				break
			}
		}
		cands = cands[:cut]
		sum = 0
		for _, c := range cands {
			sum += c.p
		}
	}
	r := s.Rng.Float64() * sum
	for _, c := range cands {
		r -= c.p
		if r <= 0 {
			return c.id
		}
	}
	return cands[len(cands)-1].id
}

// topK returns the indices of the k largest logits, sorted descending.
// Single O(vocab) pass with insertion into a k-sized sorted list.
func topK(logits []float32, k int) []int {
	idx := make([]int, 0, k+1)
	for i, l := range logits {
		if len(idx) == k && l <= logits[idx[k-1]] {
			continue
		}
		p := len(idx)
		for p > 0 && logits[idx[p-1]] < l {
			p--
		}
		idx = append(idx, 0)
		copy(idx[p+1:], idx[p:len(idx)-1])
		idx[p] = i
		if len(idx) > k {
			idx = idx[:k]
		}
	}
	return idx
}

// Stats accumulates timing for the last Generate call.
type Stats struct {
	PrefillToks int
	Prefill     time.Duration
	DecodeToks  int
	Decode      time.Duration
}

// Generate runs prefill on prompt then decodes until EOS, EOS2 or maxTok.
// Prefill is done in chunks so weights are read once per chunk, not per token.
// onToken, if non-nil, is called with each generated token id.
func (m *Model) Generate(prompt []int, s *Sampler, maxTok int, onToken func(id int)) ([]int, error) {
	m.Stats = Stats{}
	// one big prefill batch: weights are dequantized once per chunk, so
	// larger chunks amortize dequant over more tokens. Cap keeps scratch
	// memory bounded.
	chunk := min(1024, m.Cfg.MaxCtx-m.pos)
	if chunk <= 0 {
		return nil, fmt.Errorf("context length exceeded")
	}
	for i := 0; i < len(prompt); i += chunk {
		end := min(i+chunk, len(prompt))
		t0 := time.Now()
		logits, err := m.Forward(prompt[i:end])
		m.Stats.Prefill += time.Since(t0)
		m.Stats.PrefillToks += end - i
		if err != nil {
			return nil, err
		}
		if end == len(prompt) {
			return m.decode(logits, s, maxTok, onToken)
		}
	}
	return nil, nil
}

func (m *Model) decode(logits []float32, s *Sampler, maxTok int, onToken func(id int)) ([]int, error) {
	var out []int
	for len(out) < maxTok {
		id := s.Sample(logits)
		if id == m.Cfg.EOS || id == m.Cfg.EOS2 {
			break
		}
		out = append(out, id)
		if onToken != nil {
			onToken(id)
		}
		t0 := time.Now()
		var err error
		logits, err = m.Forward([]int{id})
		m.Stats.Decode += time.Since(t0)
		m.Stats.DecodeToks++
		if err != nil {
			return out, err
		}
	}
	return out, nil
}
