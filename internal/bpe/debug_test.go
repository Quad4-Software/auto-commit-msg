// SPDX-License-Identifier: 0BSD

package bpe

import (
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

func TestProbe(t *testing.T) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	tokens := f.Strs("tokenizer.ggml.tokens")
	tk := New(tokens, f.Strs("tokenizer.ggml.merges"), 151643, 151645, nil)
	for _, s := range []string{"Hello", "Ġworld", "Ġ", "world", "Ġthe", "def", "Ġdef"} {
		t.Logf("vocab[%q] = %v", s, tk.vocab[s])
	}
	t.Logf("token[9707]=%q token[220]=%q token[1917]=%q", tokens[9707], tokens[220], tokens[1917])
	// simulate bpe on "Ġworld"
	t.Log("bpe(Ġworld) =", tk.bpe("Ġworld"))
	t.Log("bpe(Hello) =", tk.bpe("Hello"))
	t.Log("split:", split("Hello world"))
	for _, p := range split("Hello world") {
		t.Logf("piece %q enc %q ids %v", p, tk.encodeBytes(p), tk.bpe(tk.encodeBytes(p)))
	}
}
