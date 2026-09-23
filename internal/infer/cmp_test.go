// SPDX-License-Identifier: 0BSD

package infer

import (
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

// TestAsmVsScalarLogits checks the AVX2 dequant path produces bit-identical
// logits to the scalar path on a 200-token prompt (exercises every quantized
// tensor across all layers).
func TestAsmVsScalarLogits(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	prompt := make([]int, 200)
	for i := range prompt {
		prompt[i] = 1000 + i*37%60000
	}

	useAsmDequant = false
	m1, _ := Load(f, 512)
	l1, _ := m1.Forward(prompt)

	useAsmDequant = true
	m2, _ := Load(f, 512)
	l2, _ := m2.Forward(prompt)

	for i := range l1 {
		if l1[i] != l2[i] {
			t.Fatalf("logit %d differs: %v vs %v", i, l1[i], l2[i])
		}
	}
}
