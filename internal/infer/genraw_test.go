package infer

import (
	"os"
	"strings"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

// TestGenRaw prints raw generation for a given prompt, to debug new models.
// ACM_TEST_MODEL=model.gguf ACM_RAW_PROMPT="..." go test -run TestGenRaw -v
func TestGenRaw(t *testing.T) {
	path := os.Getenv("ACM_TEST_MODEL")
	if path == "" {
		path = "../../model.gguf"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skip("no model")
	}
	f, err := gguf.Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	tk := tok(t, f)
	m, err := Load(f, 2048)
	if err != nil {
		t.Fatal(err)
	}
	prompt := os.Getenv("ACM_RAW_PROMPT")
	if prompt == "" {
		prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
			"You write git commit messages. Reply with a single concise subject line." +
			"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
			strings.Repeat("diff --git a/x.go b/x.go\n+func foo() {}\n", 2) +
			"<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	}
	ids := tk.Encode(prompt)
	t.Log("ids:", ids[:min(15, len(ids))])
	gen, err := m.Generate(ids, &Sampler{Temp: 0}, 40, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("raw gen ids: %v", gen)
	t.Logf("raw text: %q", tk.Decode(gen))
}
