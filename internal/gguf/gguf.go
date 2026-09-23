// SPDX-License-Identifier: 0BSD

// Package gguf parses GGUF v3 model files: metadata, tensor index and the
// raw tensor data blob.
package gguf

import (
	"encoding/binary"
	"fmt"
	"math"
)

// Metadata value types.
const (
	tUint8   = 0
	tInt8    = 1
	tUint16  = 2
	tInt16   = 3
	tUint32  = 4
	tInt32   = 5
	tFloat32 = 6
	tBool    = 7
	tString  = 8
	tArray   = 9
	tUint64  = 10
	tInt64   = 11
	tFloat64 = 12
)

// ggml tensor types we can dequantize.
const (
	TypeF32  = 0
	TypeF16  = 1
	TypeQ4_0 = 2
	TypeQ5_0 = 6
	TypeQ8_0 = 8
	TypeQ4_K = 12
	TypeQ5_K = 13
	TypeQ6_K = 14
)

type Tensor struct {
	Name   string
	Dims   []uint64 // dims[0] is the contiguous (innermost) dimension
	Type   uint32
	Offset uint64 // absolute offset into Data
	Numel  int
	Data   []byte
}

// RowBytes returns the byte size of one row (one outer dimension slice).
func (t *Tensor) RowBytes() int {
	if len(t.Dims) == 0 {
		return len(t.Data)
	}
	n := int(t.Dims[0])
	bs, ts := BlockSize(t.Type)
	return n / bs * ts
}

// Rows returns the number of rows (elements / innermost dim).
func (t *Tensor) Rows() int {
	if len(t.Dims) < 2 {
		return 1
	}
	return t.Numel / int(t.Dims[0])
}

// BlockSize returns (elements per block, bytes per block) for a type.
func BlockSize(typ uint32) (bs, ts int) {
	switch typ {
	case TypeF32:
		return 1, 4
	case TypeF16:
		return 1, 2
	case TypeQ4_0:
		return 32, 18
	case TypeQ5_0:
		return 32, 22
	case TypeQ8_0:
		return 32, 34
	case TypeQ4_K:
		return 256, 144
	case TypeQ5_K:
		return 256, 176
	case TypeQ6_K:
		return 256, 210
	}
	return 0, 0
}

func TypeName(typ uint32) string {
	switch typ {
	case TypeF32:
		return "f32"
	case TypeF16:
		return "f16"
	case TypeQ4_0:
		return "q4_0"
	case TypeQ5_0:
		return "q5_0"
	case TypeQ8_0:
		return "q8_0"
	case TypeQ4_K:
		return "q4_k"
	case TypeQ5_K:
		return "q5_k"
	case TypeQ6_K:
		return "q6_k"
	}
	return fmt.Sprintf("type%d", typ)
}

type File struct {
	Version uint32
	KV      map[string]any
	Tensors map[string]*Tensor
	Order   []string
}

func Parse(data []byte) (*File, error) {
	r := &reader{b: data}
	if string(r.take(4)) != "GGUF" {
		return nil, fmt.Errorf("not a GGUF file")
	}
	f := &File{
		Version: r.u32(),
		KV:      map[string]any{},
		Tensors: map[string]*Tensor{},
	}
	if f.Version != 3 && f.Version != 2 {
		return nil, fmt.Errorf("unsupported GGUF version %d", f.Version)
	}
	nTensors := r.u64()
	nKV := r.u64()
	for i := uint64(0); i < nKV; i++ {
		k := r.str()
		f.KV[k] = r.value()
	}
	for i := uint64(0); i < nTensors; i++ {
		name := r.str()
		nd := r.u32()
		dims := make([]uint64, nd)
		numel := 1
		for j := range dims {
			dims[j] = r.u64()
			numel *= int(dims[j])
		}
		typ := r.u32()
		off := r.u64()
		t := &Tensor{Name: name, Dims: dims, Type: typ, Offset: off, Numel: numel}
		f.Tensors[name] = t
		f.Order = append(f.Order, name)
	}
	align := uint64(32)
	if v, ok := f.KV["general.alignment"]; ok {
		align = toU64(v)
	}
	dataStart := (uint64(r.pos) + align - 1) / align * align
	for _, name := range f.Order {
		t := f.Tensors[name]
		bs, ts := BlockSize(t.Type)
		if bs == 0 {
			return nil, fmt.Errorf("tensor %s: unsupported type %d", name, t.Type)
		}
		size := uint64(t.Numel/bs) * uint64(ts)
		abs := dataStart + t.Offset
		if abs+size > uint64(len(data)) {
			return nil, fmt.Errorf("tensor %s: out of bounds", name)
		}
		t.Data = data[abs : abs+size : abs+size]
	}
	return f, nil
}

func toU64(v any) uint64 {
	switch n := v.(type) {
	case uint8:
		return uint64(n)
	case int8:
		return uint64(int64(n))
	case int16:
		return uint64(int64(n))
	case uint16:
		return uint64(n)
	case uint32:
		return uint64(n)
	case uint64:
		return n
	case int32:
		return uint64(n)
	case int64:
		return uint64(n)
	}
	return 0
}

// Str returns a string metadata value.
func (f *File) Str(key string) string {
	if v, ok := f.KV[key].(string); ok {
		return v
	}
	return ""
}

// U64 returns an integer metadata value.
func (f *File) U64(key string, def uint64) uint64 {
	if v, ok := f.KV[key]; ok {
		return toU64(v)
	}
	return def
}

// F32 returns a float metadata value.
func (f *File) F32(key string, def float32) float32 {
	switch v := f.KV[key].(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	}
	return def
}

// Strs returns a []string metadata value.
func (f *File) Strs(key string) []string {
	arr, _ := f.KV[key].([]any)
	out := make([]string, len(arr))
	for i, v := range arr {
		out[i], _ = v.(string)
	}
	return out
}

// Ints returns an integer-array metadata value.
func (f *File) Ints(key string) []int {
	arr, _ := f.KV[key].([]any)
	out := make([]int, len(arr))
	for i, v := range arr {
		out[i] = int(toU64(v))
	}
	return out
}

// F32s returns a []float32 metadata value.
func (f *File) F32s(key string) []float32 {
	arr, _ := f.KV[key].([]any)
	out := make([]float32, len(arr))
	for i, v := range arr {
		switch n := v.(type) {
		case float32:
			out[i] = n
		case float64:
			out[i] = float32(n)
		}
	}
	return out
}

type reader struct {
	b   []byte
	pos int
}

func (r *reader) take(n int) []byte {
	b := r.b[r.pos : r.pos+n]
	r.pos += n
	return b
}

func (r *reader) u32() uint32 { return binary.LittleEndian.Uint32(r.take(4)) }
func (r *reader) u64() uint64 { return binary.LittleEndian.Uint64(r.take(8)) }

func (r *reader) str() string {
	n := r.u64()
	return string(r.take(int(n)))
}

func (r *reader) value() any {
	typ := r.u32()
	switch typ {
	case tUint8:
		return r.take(1)[0]
	case tInt8:
		return int8(r.take(1)[0])
	case tUint16:
		return uint16(binary.LittleEndian.Uint16(r.take(2)))
	case tInt16:
		return int16(binary.LittleEndian.Uint16(r.take(2)))
	case tUint32:
		return r.u32()
	case tInt32:
		return int32(r.u32())
	case tUint64:
		return r.u64()
	case tInt64:
		return int64(r.u64())
	case tFloat32:
		return math.Float32frombits(r.u32())
	case tFloat64:
		return math.Float64frombits(r.u64())
	case tBool:
		return r.take(1)[0] != 0
	case tString:
		return r.str()
	case tArray:
		elem := r.u32()
		n := r.u64()
		out := make([]any, n)
		for i := range out {
			out[i] = r.valueTyped(elem)
		}
		return out
	}
	return nil
}

// valueTyped reads a value of a known type (array elements carry no tag).
func (r *reader) valueTyped(typ uint32) any {
	switch typ {
	case tUint8:
		return r.take(1)[0]
	case tInt8:
		return int8(r.take(1)[0])
	case tUint16:
		return uint16(binary.LittleEndian.Uint16(r.take(2)))
	case tInt16:
		return int16(binary.LittleEndian.Uint16(r.take(2)))
	case tUint32:
		return r.u32()
	case tInt32:
		return int32(r.u32())
	case tUint64:
		return r.u64()
	case tInt64:
		return int64(r.u64())
	case tFloat32:
		return math.Float32frombits(r.u32())
	case tFloat64:
		return math.Float64frombits(r.u64())
	case tBool:
		return r.take(1)[0] != 0
	case tString:
		return r.str()
	}
	return nil
}
