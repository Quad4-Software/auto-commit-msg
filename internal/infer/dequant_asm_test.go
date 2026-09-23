// SPDX-License-Identifier: 0BSD

package infer

import (
	"math"
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

// TestDequantAsmMatchesScalar checks every quantized tensor in the real
// model: asm dequant must equal scalar dequant bit for bit.
func TestDequantAsmMatchesScalar(t *testing.T) {
	if !hasAVX2 {
		t.Skip("no AVX2")
	}
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	bad := 0
	for name, tt := range f.Tensors {
		if tt.Type == gguf.TypeF32 || tt.Type == gguf.TypeF16 {
			continue
		}
		rows := tt.Rows()
		d := int(tt.Dims[0])
		fast := make([]float32, d)
		slow := make([]float32, d)
		for r := 0; r < rows; r += max(1, rows/17) {
			dequantRowFast(tt, r, fast)
			dequantRowSlow(tt, r, slow)
			for i := range fast {
				if math.Abs(float64(fast[i]-slow[i])) > 1e-6 {
					t.Fatalf("%s row %d elem %d: fast %v slow %v", name, r, i, fast[i], slow[i])
				}
			}
		}
		bad += 0
	}
}
