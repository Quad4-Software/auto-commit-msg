// SPDX-License-Identifier: 0BSD

package gguf

import (
	"os"
	"testing"
)

func TestInspect(t *testing.T) {
	path := os.Getenv("ACM_TEST_MODEL")
	if path == "" {
		path = "../../model.gguf"
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Skip("no model:", err)
	}
	f, err := Parse(data)
	if err != nil {
		t.Fatal(err)
	}
	arch := f.Str("general.architecture")
	p := arch + "."
	t.Log("arch:", arch)
	t.Log("name:", f.Str("general.basename"), f.Str("general.name"))
	t.Log("pre:", f.Str("tokenizer.ggml.pre"))
	t.Log("model:", f.Str("tokenizer.ggml.model"))
	t.Log("eos:", f.U64("tokenizer.ggml.eos_token_id", 0))
	t.Log("bos:", f.U64("tokenizer.ggml.bos_token_id", 0))
	t.Log("add_bos:", f.KV["tokenizer.ggml.add_bos_token"])
	t.Log("embd:", f.U64(p+"embedding_length", 0))
	t.Log("layers:", f.U64(p+"block_count", 0))
	t.Log("heads:", f.U64(p+"attention.head_count", 0))
	t.Log("kv heads:", f.U64(p+"attention.head_count_kv", 0))
	t.Log("rope base:", f.F32(p+"rope.freq_base", 0))
	t.Log("rope dim:", f.U64(p+"rope.dimension_count", 0))
	t.Log("rope scaling:", f.Str(p+"rope.scaling.type"),
		f.F32(p+"rope.scaling.factor", 0),
		f.F32(p+"rope.scaling.low_freq_factor", 0),
		f.F32(p+"rope.scaling.high_freq_factor", 0),
		f.U64(p+"rope.scaling.original_context_length", 0))
	for _, k := range []string{"tokenizer.ggml.token_type", "tokenizer.ggml.add_bos_token"} {
		if arr, ok := f.KV[k].([]any); ok {
			t.Log(k, "array len", len(arr), "first:", arr[:10])
		} else {
			t.Log(k, "=", f.KV[k])
		}
	}
	// find special token ids
	for i, s := range f.Strs("tokenizer.ggml.tokens") {
		if len(s) > 2 && s[0] == '<' && s[len(s)-1] == '>' {
			t.Logf("special %d %q", i, s)
			if i > 152000 {
				break
			}
		}
	}
	for _, name := range []string{"token_embd.weight", "blk.0.attn_q.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_norm.weight", "output.weight", "output_norm.weight"} {
		if tt := f.Tensors[name]; tt != nil {
			t.Logf("tensor %s dims=%v type=%s numel=%d", name, tt.Dims, TypeName(tt.Type), tt.Numel)
		} else {
			t.Logf("tensor %s MISSING", name)
		}
	}
}
