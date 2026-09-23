// SPDX-License-Identifier: 0BSD

// Package infer runs decoder-only transformer inference (qwen2/qwen3
// architecture) on quantized GGUF weights, in pure Go.
package infer

import (
	"fmt"
	"math"

	"quad4.io/auto-commit-msg/internal/gguf"
)

type Config struct {
	Arch     string
	NLayers  int
	NEmbd    int
	NHead    int
	NHeadKV  int
	HeadDim  int
	NFF      int
	Vocab    int
	MaxCtx   int
	RopeDim  int
	RopeBase float32
	RMSEps   float32
	BOS      int
	EOS      int
	EOS2     int // secondary stop token (e.g. <|endoftext|>)
}

type layer struct {
	q, k, v, o     *gguf.Tensor
	qBias, kBias   *gguf.Tensor // qwen2 only
	vBias          *gguf.Tensor
	gate, up, down *gguf.Tensor
	attnNW, ffnNW  []float32 // dequantized norms
	qNW, kNW       []float32 // qwen3 QK-norm, may be nil
}

type Model struct {
	Cfg    Config
	embd   *gguf.Tensor
	out    *gguf.Tensor // tied to embd when absent
	outNW  []float32
	layers []layer

	kc, vc [][]float32 // [layer][pos][kvdim], grown on demand
	pos    int
	freqs  []float32 // ropeFreqs(RopeDim, RopeBase)

	scratch scratchBufs // reused activation buffers, sized for max batch
	Stats   Stats
}

type scratchBufs struct {
	x, xb, h, q, k, v, att, hb, hb2 []float32
	hf, logits                      []float32
	n                               int // rows the buffers currently hold
}

// scratchFor returns activation buffers for nTok rows, growing when needed.
func (s *scratchBufs) for_(nTok, embd, qd, kvd, nff int) {
	if s.n >= nTok {
		return
	}
	s.x = make([]float32, nTok*embd)
	s.xb = make([]float32, nTok*embd)
	s.h = make([]float32, nTok*embd)
	s.q = make([]float32, nTok*qd)
	s.k = make([]float32, nTok*kvd)
	s.v = make([]float32, nTok*kvd)
	s.att = make([]float32, nTok*qd)
	s.hb = make([]float32, nTok*nff)
	s.hb2 = make([]float32, nTok*nff)
	s.hf = make([]float32, embd)
	s.n = nTok
}

func Load(f *gguf.File, maxCtx int) (*Model, error) {
	arch := f.Str("general.architecture")
	if arch != "qwen3" && arch != "qwen2" {
		return nil, fmt.Errorf("unsupported architecture %q (want qwen2 or qwen3)", arch)
	}
	p := arch + "."
	c := Config{
		Arch:     arch,
		NLayers:  int(f.U64(p+"block_count", 0)),
		NEmbd:    int(f.U64(p+"embedding_length", 0)),
		NHead:    int(f.U64(p+"attention.head_count", 0)),
		NHeadKV:  int(f.U64(p+"attention.head_count_kv", 0)),
		NFF:      int(f.U64(p+"feed_forward_length", 0)),
		MaxCtx:   int(f.U64(p+"context_length", 0)),
		RopeBase: f.F32(p+"rope.freq_base", 10000),
		RMSEps:   f.F32(p+"attention.layer_norm_rms_epsilon", 1e-6),
		Vocab:    len(f.Strs("tokenizer.ggml.tokens")),
		BOS:      int(f.U64("tokenizer.ggml.bos_token_id", 0)),
		EOS:      int(f.U64("tokenizer.ggml.eos_token_id", 0)),
	}
	if n := int(f.U64(p+"attention.key_length", 0)); n > 0 {
		c.HeadDim = n
	} else if t := f.Tensors["blk.0.attn_q.weight"]; t != nil && len(t.Dims) > 1 {
		c.HeadDim = int(t.Dims[1]) / c.NHead
	} else {
		c.HeadDim = c.NEmbd / c.NHead
	}
	c.RopeDim = int(f.U64(p+"rope.dimension_count", uint64(c.HeadDim)))
	if c.HeadKV() == 0 || c.NEmbd == 0 || c.NLayers == 0 {
		return nil, fmt.Errorf("missing required metadata for arch %s", arch)
	}
	if maxCtx <= 0 || maxCtx > c.MaxCtx {
		maxCtx = c.MaxCtx
	}
	c.MaxCtx = maxCtx
	c.EOS2 = -1
	if arch == "qwen3" || arch == "qwen2" {
		// <|endoftext|> and <|im_end|> are the two stop tokens for chat models
		for i, tok := range f.Strs("tokenizer.ggml.tokens") {
			if tok == "<|endoftext|>" {
				c.EOS2 = i
			}
		}
	}

	get := func(name string) *gguf.Tensor { return f.Tensors[name] }
	m := &Model{
		Cfg:   c,
		embd:  get("token_embd.weight"),
		out:   get("output.weight"),
		kc:    make([][]float32, c.NLayers),
		vc:    make([][]float32, c.NLayers),
		freqs: ropeFreqs(c.RopeDim, c.RopeBase),
	}
	if on := get("output_norm.weight"); on != nil {
		m.outNW = normWeight(on)
	}
	if m.embd == nil || m.outNW == nil {
		return nil, fmt.Errorf("missing embedding or output norm tensors")
	}
	if m.out == nil {
		m.out = m.embd // tied embeddings
	}
	for i := 0; i < c.NLayers; i++ {
		b := fmt.Sprintf("blk.%d.", i)
		l := layer{
			q:     get(b + "attn_q.weight"),
			k:     get(b + "attn_k.weight"),
			v:     get(b + "attn_v.weight"),
			o:     get(b + "attn_output.weight"),
			qBias: get(b + "attn_q.bias"),
			kBias: get(b + "attn_k.bias"),
			vBias: get(b + "attn_v.bias"),
			gate:  get(b + "ffn_gate.weight"),
			up:    get(b + "ffn_up.weight"),
			down:  get(b + "ffn_down.weight"),
		}
		if an := get(b + "attn_norm.weight"); an != nil {
			l.attnNW = normWeight(an)
		}
		if fn := get(b + "ffn_norm.weight"); fn != nil {
			l.ffnNW = normWeight(fn)
		}
		if qn := get(b + "attn_q_norm.weight"); qn != nil {
			l.qNW = normWeight(qn)
		}
		if kn := get(b + "attn_k_norm.weight"); kn != nil {
			l.kNW = normWeight(kn)
		}
		if l.attnNW == nil || l.q == nil || l.k == nil || l.v == nil ||
			l.o == nil || l.ffnNW == nil || l.gate == nil || l.up == nil || l.down == nil {
			return nil, fmt.Errorf("layer %d: missing tensors", i)
		}
		m.layers = append(m.layers, l)
	}
	return m, nil
}

func (c *Config) HeadKV() int {
	if c.NHeadKV == 0 {
		return c.NHead
	}
	return c.NHeadKV
}
func (c *Config) kvDim() int { return c.HeadKV() * c.HeadDim }

func rmsnorm(dst, x, w []float32, eps float32) {
	var ss float32
	for _, v := range x {
		ss += v * v
	}
	ss = 1 / float32(math.Sqrt(float64(ss/float32(len(x)))+float64(eps)))
	for i := range dst {
		dst[i] = x[i] * ss * w[i]
	}
}

func normWeight(t *gguf.Tensor) []float32 {
	// norm tensors are small and always f32 in practice
	out := make([]float32, t.Numel)
	dequantRow(t, 0, out)
	return out
}

// DebugLayer, if non-nil, is called with (layer index, x buffer) after each
// layer's residual update. x is [nTok][embd]. Used by tests.
var DebugLayer func(li int, x []float32)

// rope applies rotary embeddings to one head vector at position pos.
// freqs[i] = base^(-2i/dim), precomputed once per model.
func rope(x []float32, pos int, dim int, freqs []float32) {
	half := dim / 2
	for i := 0; i < half; i++ {
		c, s := float32(math.Cos(float64(float32(pos)*freqs[i]))), float32(math.Sin(float64(float32(pos)*freqs[i])))
		a, b := x[i], x[i+half]
		x[i] = a*c - b*s
		x[i+half] = a*s + b*c
	}
}

func ropeFreqs(dim int, base float32) []float32 {
	f := make([]float32, dim/2)
	for i := range f {
		f[i] = 1 / float32(math.Pow(float64(base), float64(2*i)/float64(dim)))
	}
	return f
}

// fastExp approximates e^x for x <= 0 to ~1e-6 relative accuracy, plenty for
// softmax and SiLU. exp(x) = 2^(x*log2e) = 2^i * 2^f; the fractional part is
// evaluated with a degree-5 minimax polynomial for 2^f on [0,1).
func fastExp(x float32) float32 {
	if x < -87 {
		return 0
	}
	t := x * 1.4426950408889634 // log2(e)
	i := int32(math.Floor(float64(t)))
	f := t - float32(i)
	// 2^f polynomial
	p := 1.0000001 + f*(0.69314718+f*(0.24022650+f*(0.05550411+f*(0.00961813+f*0.00133336))))
	return float32(math.Ldexp(float64(p), int(i)))
}

func softmax(x []float32) {
	mx := x[0]
	for _, v := range x {
		if v > mx {
			mx = v
		}
	}
	var sum float32
	for i, v := range x {
		e := fastExp(v - mx)
		x[i] = e
		sum += e
	}
	for i := range x {
		x[i] /= sum
	}
}

// Forward evaluates a batch of tokens at consecutive positions starting at
// m.pos. Returns logits for the LAST token only (callers only need those).
func (m *Model) Forward(tokens []int) ([]float32, error) {
	c := &m.Cfg
	nTok := len(tokens)
	if m.pos+nTok > c.MaxCtx {
		return nil, fmt.Errorf("context length exceeded (%d > %d)", m.pos+nTok, c.MaxCtx)
	}
	embd, nh, nkv, hd := c.NEmbd, c.NHead, c.HeadKV(), c.HeadDim
	kvd := nkv * hd
	qd := nh * hd

	m.scratch.for_(nTok, embd, qd, kvd, c.NFF)
	x := m.scratch.x[:nTok*embd]
	xb := m.scratch.xb[:nTok*embd]
	h := m.scratch.h[:nTok*embd]
	q := m.scratch.q[:nTok*qd]
	k := m.scratch.k[:nTok*kvd]
	v := m.scratch.v[:nTok*kvd]
	att := m.scratch.att[:nTok*qd]
	hb := m.scratch.hb[:nTok*c.NFF]
	hb2 := m.scratch.hb2[:nTok*c.NFF]

	for j, tok := range tokens {
		dequantRow(m.embd, tok, x[j*embd:(j+1)*embd])
	}

	for li, l := range m.layers {
		for j := 0; j < nTok; j++ {
			rmsnorm(h[j*embd:(j+1)*embd], x[j*embd:(j+1)*embd], l.attnNW, c.RMSEps)
		}
		matmul(q, l.q, h, nTok)
		matmul(k, l.k, h, nTok)
		matmul(v, l.v, h, nTok)
		addBias(q, l.qBias)
		addBias(k, l.kBias)
		addBias(v, l.vBias)

		// grow the KV cache for this layer in blocks of 256 positions
		need := (m.pos + nTok) * kvd
		kcl, vcl := m.kc[li], m.vc[li]
		for len(kcl) < need {
			kcl = append(kcl, make([]float32, 256*kvd)...)
			vcl = append(vcl, make([]float32, 256*kvd)...)
		}
		m.kc[li], m.vc[li] = kcl, vcl
		kc, vc := kcl[:need], vcl[:need]
		for j := 0; j < nTok; j++ {
			pos := m.pos + j
			qj := q[j*qd : j*qd+qd]
			kj := k[j*kvd : j*kvd+kvd]
			for head := 0; head < nh; head++ {
				hq := qj[head*hd : head*hd+hd]
				if l.qNW != nil {
					rmsnorm(hq, hq, l.qNW, c.RMSEps)
				}
				rope(hq, pos, c.RopeDim, m.freqs)
			}
			for head := 0; head < nkv; head++ {
				hk := kj[head*hd : head*hd+hd]
				if l.kNW != nil {
					rmsnorm(hk, hk, l.kNW, c.RMSEps)
				}
				rope(hk, pos, c.RopeDim, m.freqs)
			}
			copy(kc[pos*kvd:], kj)
			copy(vc[pos*kvd:], v[j*kvd:j*kvd+kvd])
		}

		scale := 1 / float32(math.Sqrt(float64(hd)))
		parFor(nTok*nh, func(lo, hi int) {
			score := make([]float32, 0, 64)
			for idx := lo; idx < hi; idx++ {
				j, head := idx/nh, idx%nh
				pos := m.pos + j
				qh := q[j*qd+head*hd : j*qd+(head+1)*hd]
				kvh := head / (nh / nkv)
				n := pos + 1
				if cap(score) < n {
					score = make([]float32, n)
				}
				score = score[:n]
				kbase := kvh * hd
				for p := 0; p <= pos; p++ {
					score[p] = dot(qh, kc[p*kvd+kbase:p*kvd+kbase+hd]) * scale
				}
				softmax(score)
				out := att[j*qd+head*hd : j*qd+(head+1)*hd]
				clear(out)
				for p := 0; p <= pos; p++ {
					axpy(out, score[p], vc[p*kvd+kbase:p*kvd+kbase+hd])
				}
			}
		})

		matmul(xb, l.o, att, nTok)
		for i := range x {
			x[i] += xb[i]
		}

		for j := 0; j < nTok; j++ {
			rmsnorm(h[j*embd:(j+1)*embd], x[j*embd:(j+1)*embd], l.ffnNW, c.RMSEps)
		}
		matmul(hb, l.gate, h, nTok)
		matmul(hb2, l.up, h, nTok)
		for i, g := range hb {
			hb[i] = g * (1 / (1 + fastExp(-g))) * hb2[i]
		}
		matmul(xb, l.down, hb, nTok)
		for i := range x {
			x[i] += xb[i]
		}
		if DebugLayer != nil {
			DebugLayer(li, x)
		}
	}
	m.pos += nTok

	last := x[(nTok-1)*embd : nTok*embd]
	hf := m.scratch.hf
	rmsnorm(hf, last, m.outNW, c.RMSEps)
	if cap(m.scratch.logits) < c.Vocab {
		m.scratch.logits = make([]float32, c.Vocab)
	}
	logits := m.scratch.logits[:c.Vocab]
	matvec(logits, m.out, hf)
	return logits, nil
}

func addBias(dst []float32, t *gguf.Tensor) {
	if t == nil {
		return
	}
	b := make([]float32, t.Numel)
	dequantRow(t, 0, b)
	for i := range dst {
		dst[i] += b[i%len(b)]
	}
}

// Pos returns the current sequence position.
func (m *Model) Pos() int { return m.pos }

// Reset rewinds the sequence position so a fresh prompt can reuse the model.
// Stale KV cache entries are overwritten as new tokens are appended.
func (m *Model) Reset() { m.pos = 0 }

// MaxLogit returns the argmax token id for a logits slice.
func MaxLogit(logits []float32) int {
	best := 0
	for i := 1; i < len(logits); i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}
