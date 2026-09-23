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
	tk := New(tokens, f.Strs("tokenizer.ggml.merges"),
		int(f.U64("tokenizer.ggml.bos_token_id", 0)),
		int(f.U64("tokenizer.ggml.eos_token_id", 0)), nil,
		f.Str("tokenizer.ggml.pre"))
	for _, s := range []string{"Hello", "Ġworld", "Ġ", "world", "Ġthe", "def", "Ġdef"} {
		t.Logf("vocab[%q] = %v", s, tk.vocab[s])
	}
	t.Logf("token[9707]=%q token[220]=%q token[1917]=%q", tokens[9707], tokens[220], tokens[1917])
	// simulate bpe on "Ġworld"
	t.Log("bpe(Ġworld) =", tk.bpe("Ġworld"))
	t.Log("bpe(Hello) =", tk.bpe("Hello"))
	t.Log("split:", tk.split("Hello world"))
	for _, p := range tk.split("Hello world") {
		t.Logf("piece %q enc %q ids %v", p, tk.encodeBytes(p), tk.bpe(tk.encodeBytes(p)))
	}
}
