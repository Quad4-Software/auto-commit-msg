// SPDX-License-Identifier: 0BSD

package infer

import (
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

func BenchmarkMatvecSizes(b *testing.B) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		b.Skip("no model")
	}
	f, _ := gguf.Parse(data)
	for _, name := range []string{"output.weight", "blk.0.ffn_down.weight", "blk.0.attn_q.weight"} {
		w := f.Tensors[name]
		if w == nil {
			continue
		}
		x := make([]float32, w.Dims[0])
		dst := make([]float32, w.Rows())
		matvec(dst, w, x) // warm
		b.Run(name, func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				matvec(dst, w, x)
			}
			b.ReportMetric(float64(len(w.Data))*float64(b.N)/b.Elapsed().Seconds()/1e9, "GB/s")
		})
	}
}
