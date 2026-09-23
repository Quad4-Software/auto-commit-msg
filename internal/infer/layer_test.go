// SPDX-License-Identifier: 0BSD

package infer

import (
	"math"
	"testing"
)

func TestLayers(t *testing.T) {
	f, m := loadTest(t)
	tk := tok(t, f)
	ids := tk.Encode("The capital of France is")
	DebugLayer = func(li int, x []float32) {
		embd := m.Cfg.NEmbd
		last := x[(len(x)/embd-1)*embd:]
		var mx float32
		nan := 0
		for _, v := range last {
			if math.IsNaN(float64(v)) {
				nan++
				continue
			}
			if float32(math.Abs(float64(v))) > mx {
				mx = float32(math.Abs(float64(v)))
			}
		}
		t.Logf("layer %2d last-pos max|x|=%.4g nan=%d", li, mx, nan)
	}
	defer func() { DebugLayer = nil }()
	if _, err := m.Forward(ids); err != nil {
		t.Fatal(err)
	}
}
