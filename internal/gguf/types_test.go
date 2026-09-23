package gguf

import (
	"os"
	"testing"
)

// TestTypeDist logs the tensor type distribution of the test model.
func TestTypeDist(t *testing.T) {
	data, err := os.ReadFile("../../model.gguf")
	if err != nil {
		t.Skip("no model.gguf")
	}
	f, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	cnt := map[uint32]int{}
	var bytes map[uint32]int64 = map[uint32]int64{}
	for _, tn := range f.Tensors {
		cnt[tn.Type]++
		bytes[tn.Type] += int64(len(tn.Data))
	}
	t.Logf("types=%v bytes=%v", cnt, bytes)
}
