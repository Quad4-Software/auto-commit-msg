package infer

import (
	"os"
	"strings"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

// Bisects the position where llama3.2 logits diverge: pad "The capital of
// France is" with N filler tokens and check the top logit stays " Paris".
func TestPosBisect(t *testing.T) {
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
	paris := tk.Encode(" Paris")[0]
	for _, pad := range []int{0, 2, 4, 8, 16, 32, 64, 128, 256} {
		m.Reset()
		prompt := "<|begin_of_text|>" + strings.Repeat(" the", pad) + " The capital of France is"
		ids := tk.Encode(prompt)
		logits, err := m.Forward(ids)
		if err != nil {
			t.Fatal(err)
		}
		top := MaxLogit(logits)
		t.Logf("pad=%3d ntok=%3d top=%q(%d) paris_logit=%.3f top_logit=%.3f",
			pad, len(ids), tk.Decode([]int{top}), top, logits[paris], logits[top])
	}
}
