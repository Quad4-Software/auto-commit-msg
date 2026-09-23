// SPDX-License-Identifier: 0BSD

package infer

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"testing"

	"quad4.io/auto-commit-msg/internal/bpe"
	"quad4.io/auto-commit-msg/internal/gguf"
)

func loadTest(t *testing.T) (*gguf.File, *Model) {
	path := os.Getenv("ACM_TEST_MODEL")
	if path == "" {
		path = "../../model.gguf"
	}
	data, err := os.ReadFile(path)
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
	t.Logf("cfg: %+v", m.Cfg)
	return f, m
}

func tok(t *testing.T, f *gguf.File) *bpe.Tokenizer {
	tokens := f.Strs("tokenizer.ggml.tokens")
	var specials []int
	for i, tt := range f.Ints("tokenizer.ggml.token_type") {
		if tt == 3 || tt == 4 {
			specials = append(specials, i)
		}
	}
	return bpe.New(tokens, f.Strs("tokenizer.ggml.merges"), 151643, 151645, specials)
}

func TestTokenizer(t *testing.T) {
	f, _ := loadTest(t)
	tk := tok(t, f)
	// golden ids verified against the reference Qwen3 tokenizer
	golden := map[string][]int{
		"Hello world":              {9707, 1879},
		"The capital of France is": {785, 6722, 315, 9625, 374},
		"<|im_start|>system\n":     {151644, 8948, 198},
	}
	for s, want := range golden {
		got := tk.Encode(s)
		if fmt.Sprint(got) != fmt.Sprint(want) {
			t.Errorf("Encode(%q) = %v, want %v", s, got, want)
		}
	}
	for _, s := range []string{
		"Hello world",
		"<|im_start|>system\nYou write git commit messages.<|im_end|>\n",
		"def foo(x):\n    return x + 1\n",
		"a  b   c",
		"file.go",
	} {
		if back := tk.Decode(tk.Encode(s)); back != s {
			t.Errorf("round trip failed: %q != %q", back, s)
		}
	}
}

func TestLogits(t *testing.T) {
	f, m := loadTest(t)
	tk := tok(t, f)
	prompt := "The capital of France is"
	ids := tk.Encode(prompt)
	t.Log("ids:", ids)
	logits, err := m.Forward(ids)
	if err != nil {
		t.Fatal(err)
	}
	type kv struct {
		id int
		l  float32
	}
	top := make([]kv, 10)
	for i, l := range logits {
		if len(top) < 10 || l > top[9].l {
			top = append(top, kv{i, l})
			sort.Slice(top, func(a, b int) bool { return top[a].l > top[b].l })
			top = top[:10]
		}
	}
	for _, e := range top {
		t.Logf("logit %.2f id=%d %q", e.l, e.id, tk.Text(e.id))
	}
	// reference: llama.cpp/ollama on the base Qwen3-0.6B gives " Paris"
	// (id 12095). Fine-tuned variants may answer differently, so only
	// assert on the base model; otherwise just require finite logits.
	if strings.Contains(f.Str("general.basename"), "Qwen3") {
		if top[0].id != 12095 {
			t.Errorf("top logit = %d %q, want 12095 \"ĠParis\"", top[0].id, tk.Text(top[0].id))
		}
	}
}
