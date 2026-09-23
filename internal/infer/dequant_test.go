// SPDX-License-Identifier: 0BSD

package infer

import (
	"math"
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

func TestDequantStats(t *testing.T) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	for _, name := range []string{
		"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_k.weight",
		"blk.0.ffn_gate.weight", "blk.0.attn_norm.weight", "output.weight",
	} {
		tt := f.Tensors[name]
		if tt == nil {
			t.Logf("%s missing", name)
			continue
		}
		row := make([]float32, tt.Dims[0])
		dequantRow(tt, 0, row)
		var mn, mx, mean, sd float32
		mn, mx = row[0], row[0]
		for _, v := range row {
			mn = min(mn, v)
			mx = max(mx, v)
			mean += v
		}
		mean /= float32(len(row))
		for _, v := range row {
			sd += (v - mean) * (v - mean)
		}
		sd = float32(math.Sqrt(float64(sd / float32(len(row)))))
		t.Logf("%s type=%s dims=%v row0 min=%.4f max=%.4f mean=%.5f sd=%.5f first4=%v",
			name, gguf.TypeName(tt.Type), tt.Dims, mn, mx, mean, sd, row[:4])
	}
}
