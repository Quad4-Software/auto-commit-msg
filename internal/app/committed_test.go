// SPDX-License-Identifier: 0BSD

package app

import (
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/bpe"
	"quad4.io/auto-commit-msg/internal/gguf"
	"quad4.io/auto-commit-msg/internal/infer"
)

// TestCommittedModel is an integration check for the default model: it must
// emit a Conventional Commits line for each fixture diff. Point
// ACM_TEST_MODEL at a committed-* GGUF to enable.
func TestCommittedModel(t *testing.T) {
	modelPath := os.Getenv("ACM_TEST_MODEL")
	if modelPath == "" {
		t.Skip("set ACM_TEST_MODEL")
	}
	data, err := os.ReadFile(modelPath)
	if err != nil {
		t.Skip("no model:", err)
	}
	f, err := gguf.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	tokens := f.Strs("tokenizer.ggml.tokens")
	var specials []int
	for i, tt := range f.Ints("tokenizer.ggml.token_type") {
		if tt == 3 || tt == 4 {
			specials = append(specials, i)
		}
	}
	tok := bpe.New(tokens, f.Strs("tokenizer.ggml.merges"),
		int(f.U64("tokenizer.ggml.bos_token_id", 0)),
		int(f.U64("tokenizer.ggml.eos_token_id", 0)), specials)
	m, err := infer.Load(f, 2048)
	if err != nil {
		t.Fatal(err)
	}

	for _, name := range []string{"small", "medium", "large"} {
		diffB, err := os.ReadFile("/tmp/diffbench/" + name + ".diff")
		if err != nil {
			continue
		}
		diff := string(diffB)
		if len(tok.Encode(diff)) > 1024 {
			diff = compressDiff(diff, 1024, tok)
		}
		prompt := "<|im_start|>system\n" + committedSys + "<|im_end|>\n" +
			"<|im_start|>user\nDiff:\n" + diff + "\n\n/no_think<|im_end|>\n" +
			"<|im_start|>assistant\n<think>\n\n</think>\n\n"
		m.Reset()
		gen, err := m.Generate(tok.Encode(prompt), &infer.Sampler{Temp: 0}, 64, nil)
		if err != nil {
			t.Fatal(err)
		}
		msg := cleanup(tok.Decode(gen))
		t.Logf("%s: %q", name, msg)
		if !conventionalRe.MatchString(msg) {
			t.Errorf("%s: not conventional format: %q", name, msg)
		}
	}
}
