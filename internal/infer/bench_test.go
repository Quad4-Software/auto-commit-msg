package infer

import (
	"math/rand"
	"os"
	"testing"

	"quad4.io/auto-commit-msg/internal/gguf"
)

func BenchmarkDot(b *testing.B) {
	a := make([]float32, 1024)
	c := make([]float32, 1024)
	for i := range a {
		a[i], c[i] = rand.Float32(), rand.Float32()
	}
	b.Run("scalar", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink = dotScalar(a, c)
		}
	})
	b.Run("asm", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			sink = dot(a, c)
		}
	})
}

func BenchmarkForward1(b *testing.B) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		b.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	m, _ := Load(f, 2048)
	m.Forward([]int{9707})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward([]int{9707})
	}
}

func BenchmarkMatvecQ4K(b *testing.B) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		b.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	w := f.Tensors["blk.0.ffn_down.weight"]
	x := make([]float32, w.Dims[0])
	dst := make([]float32, w.Rows())
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		matvec(dst, w, x)
	}
}

var sink float32

func BenchmarkGenerate(b *testing.B) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		b.Skip("no model.gguf")
	}
	f, _ := gguf.Parse(data)
	m, _ := Load(f, 2048)
	s := &Sampler{Temp: 0}
	prompt := make([]int, 128)
	for i := range prompt {
		prompt[i] = 1000 + i
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Generate(prompt, s, 16, nil)
		m.Reset()
	}
}

func BenchmarkForward1Mmap(b *testing.B) {
	f2, err := os.Open("../../model.gguf")
	if err != nil {
		b.Skip("no model")
	}
	st, _ := f2.Stat()
	data, err := syscallMmap(f2, int(st.Size()))
	if err != nil {
		b.Skip(err)
	}
	f, _ := gguf.Parse(data)
	m, _ := Load(f, 2048)
	m.Forward([]int{9707})
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Forward([]int{9707})
	}
}

func BenchmarkGenerateMmap(b *testing.B) {
	f2, err := os.Open("../../model.gguf")
	if err != nil {
		b.Skip("no model")
	}
	st, _ := f2.Stat()
	data, _ := syscallMmap(f2, int(st.Size()))
	f, _ := gguf.Parse(data)
	m, _ := Load(f, 2048)
	s := &Sampler{Temp: 0}
	prompt := make([]int, 724)
	for i := range prompt {
		prompt[i] = 1000 + i%60000
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Generate(prompt, s, 10, nil)
		m.Reset()
	}
}
