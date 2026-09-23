package gguf

import (
	"fmt"
	"os"
	"sort"
	"testing"
)

func TestDumpKV(t *testing.T) {
	data, err := os.ReadFile(os.Getenv("ACM_TEST_MODEL"))
	if err != nil {
		t.Skip("no model")
	}
	f, _ := Parse(data)
	var keys []string
	for k := range f.KV {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		v := f.KV[k]
		if arr, ok := v.([]any); ok {
			t.Logf("%s = array[%d]", k, len(arr))
		} else {
			t.Logf("%s = %v", k, v)
		}
	}
	_ = fmt.Sprint()
}
