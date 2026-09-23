// SPDX-License-Identifier: 0BSD

package infer

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"

	"quad4.io/auto-commit-msg/internal/gguf"
)

// f16 decodes an IEEE 754 half-precision float.
func f16(b []byte) float32 {
	h := binary.LittleEndian.Uint16(b)
	s := uint32(h>>15) & 1
	e := uint32(h>>10) & 0x1f
	m := uint32(h) & 0x3ff
	var f float32
	switch {
	case e == 0:
		f = float32(m) * 0x1p-24
	case e == 31:
		f = float32(math.Inf(1))
		if m != 0 {
			f = float32(math.NaN())
		}
	default:
		f = float32(m) * 0x1p-10
		f += 1
		f *= float32(math.Ldexp(1, int(e)-15))
	}
	if s == 1 {
		return -f
	}
	return f
}

func f32(b []byte) float32 { return math.Float32frombits(binary.LittleEndian.Uint32(b)) }

// dequantRow dequantizes row r of tensor t into dst (len == t.Dims[0]).
var useAsmDequant = hasAVX2 && os.Getenv("ACM_NO_ASM_DEQUANT") == ""

func dequantRow(t *gguf.Tensor, r int, dst []float32) {
	if useAsmDequant {
		dequantRowFast(t, r, dst)
		return
	}
	dequantRowSlow(t, r, dst)
}

func dequantRowSlow(t *gguf.Tensor, r int, dst []float32) {
	row := t.Data[r*t.RowBytes():]
	switch t.Type {
	case gguf.TypeF32:
		for i := range dst {
			dst[i] = f32(row[i*4:])
		}
	case gguf.TypeF16:
		for i := range dst {
			dst[i] = f16(row[i*2:])
		}
	case gguf.TypeQ8_0:
		dequantQ80(row, dst)
	case gguf.TypeQ4_0:
		dequantQ40(row, dst)
	case gguf.TypeQ5_0:
		dequantQ50(row, dst)
	case gguf.TypeQ4_K:
		dequantQ4K(row, dst)
	case gguf.TypeQ5_K:
		dequantQ5K(row, dst)
	case gguf.TypeQ6_K:
		dequantQ6K(row, dst)
	default:
		panic(fmt.Sprintf("dequant: unsupported type %s", gguf.TypeName(t.Type)))
	}
}

func dequantQ80(row []byte, dst []float32) {
	for b := 0; b*32 < len(dst); b++ {
		d := f16(row)
		qs := row[2:]
		for j := 0; j < 32; j++ {
			dst[b*32+j] = d * float32(int8(qs[j]))
		}
		row = row[34:]
	}
}

func dequantQ40(row []byte, dst []float32) {
	for b := 0; b*32 < len(dst); b++ {
		d := f16(row)
		qs := row[2:]
		for j := 0; j < 16; j++ {
			dst[b*32+j] = d * float32(int(qs[j]&0x0f)-8)
			dst[b*32+j+16] = d * float32(int(qs[j]>>4)-8)
		}
		row = row[18:]
	}
}

// dequantQ50 decodes Q5_0 blocks: 32 elems per 22 bytes
// (fp16 scale, 32 high bits, 16 packed nibbles).
func dequantQ50(row []byte, dst []float32) {
	for b := 0; b*32 < len(dst); b++ {
		blk := row[b*22:]
		d := f16(blk)
		qh := binary.LittleEndian.Uint32(blk[2:])
		qs := blk[6:]
		for j := 0; j < 16; j++ {
			x0 := int(qs[j]&0x0f) | int((qh>>j)&1)<<4
			x1 := int(qs[j]>>4) | int((qh>>(j+16))&1)<<4
			dst[b*32+j] = d * float32(x0-16)
			dst[b*32+j+16] = d * float32(x1-16)
		}
	}
}

// scaleMinK4 unpacks the 6-bit scale/min pairs of a K-quant super-block.
func scaleMinK4(j int, q []byte) (d, m byte) {
	if j < 4 {
		return q[j] & 63, q[j+4] & 63
	}
	return (q[j+4] & 0x0f) | ((q[j-4] >> 6) << 4),
		(q[j+4] >> 4) | ((q[j] >> 6) << 4)
}

func dequantQ4K(row []byte, dst []float32) {
	for nb := 0; nb*256 < len(dst); nb++ {
		blk := row[nb*144:]
		d := f16(blk)
		dmin := f16(blk[2:])
		scales := blk[4:16]
		qs := blk[16:]
		y := dst[nb*256:]
		yi, is := 0, 0
		for j := 0; j < 256; j += 64 {
			q := qs[j/2 : j/2+32]
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for _, b := range q {
				y[yi] = d1*float32(b&0x0f) - m1
				yi++
			}
			for _, b := range q {
				y[yi] = d2*float32(b>>4) - m2
				yi++
			}
			is += 2
		}
	}
}

func dequantQ5K(row []byte, dst []float32) {
	for nb := 0; nb*256 < len(dst); nb++ {
		blk := row[nb*176:]
		d := f16(blk)
		dmin := f16(blk[2:])
		scales := blk[4:16]
		qh := blk[16:48]
		qs := blk[48:176]
		y := dst[nb*256:]
		yi, is := 0, 0
		u1, u2 := byte(1), byte(2)
		for j := 0; j < 256; j += 64 {
			ql := qs[j/2 : j/2+32]
			sc, m := scaleMinK4(is, scales)
			d1, m1 := d*float32(sc), dmin*float32(m)
			sc, m = scaleMinK4(is+1, scales)
			d2, m2 := d*float32(sc), dmin*float32(m)
			for l, b := range ql {
				add := float32(0)
				if qh[l]&u1 != 0 {
					add = 16
				}
				y[yi] = d1*(float32(b&0x0f)+add) - m1
				yi++
			}
			for l, b := range ql {
				add := float32(0)
				if qh[l]&u2 != 0 {
					add = 16
				}
				y[yi] = d2*(float32(b>>4)+add) - m2
				yi++
			}
			is += 2
			u1 <<= 2
			u2 <<= 2
		}
	}
}

func dequantQ6K(row []byte, dst []float32) {
	for nb := 0; nb*256 < len(dst); nb++ {
		blk := row[nb*210:]
		ql := blk[:128]
		qh := blk[128:192]
		scales := blk[192:208]
		d := f16(blk[208:])
		for n := 0; n < 256; n += 128 {
			i := n / 128
			qlb := ql[64*i:]
			qhb := qh[32*i:]
			sc := scales[8*i:]
			y := dst[nb*256+n:]
			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int8(qlb[l]&0x0f|(qhb[l]&3)<<4) - 32
				q2 := int8(qlb[l+32]&0x0f|((qhb[l]>>2)&3)<<4) - 32
				q3 := int8(qlb[l]>>4|((qhb[l]>>4)&3)<<4) - 32
				q4 := int8(qlb[l+32]>>4|((qhb[l]>>6)&3)<<4) - 32
				y[l] = d * float32(int8(sc[is])) * float32(q1)
				y[l+32] = d * float32(int8(sc[is+2])) * float32(q2)
				y[l+64] = d * float32(int8(sc[is+4])) * float32(q3)
				y[l+96] = d * float32(int8(sc[is+6])) * float32(q4)
			}
		}
	}
}

var procs = runtime.NumCPU()

// tilePool reuses dequant tile buffers across matmul calls; without it each
// parFor worker allocates a fresh ~256KB buffer per call, which churns GC.
var tilePool = sync.Pool{New: func() any { return []float32{} }}

func parFor(n int, fn func(lo, hi int)) {
	var wg sync.WaitGroup
	step := (n + procs - 1) / procs
	for p := 0; p < n; p += step {
		lo, hi := p, min(p+step, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			fn(lo, hi)
		}()
	}
	wg.Wait()
}

// matmul computes dst[m][n] = x[m][d] * w[n][d]^T where w is a (possibly
// quantized) GGUF tensor. Weight rows are processed in blocks of br:
// each block is dequantized once into a cache-resident tile, then the
// whole activation matrix is dotted against the tile. This bounds the
// x traffic to n/br passes instead of one pass per row.
func matmul(dst []float32, w *gguf.Tensor, x []float32, m int) {
	n := w.Rows()
	d := int(w.Dims[0])
	// tile keeps dequantized weights roughly inside L2
	br := 256 * 1024 / (4 * d) // floats budget ~256KB
	br = min(br, 64)
	if br < 4 {
		br = 4
	}
	parFor((n+br-1)/br, func(lo, hi int) {
		buf := tilePool.Get().([]float32)
		if cap(buf) < br*d {
			buf = make([]float32, br*d)
		}
		buf = buf[:br*d]
		defer tilePool.Put(buf)
		for b0 := lo * br; b0 < hi*br && b0 < n; b0 += br {
			rows := min(br, n-b0)
			for r := 0; r < rows; r++ {
				dequantRow(w, b0+r, buf[r*d:(r+1)*d])
			}
			j := 0
			for ; j+4 <= m; j += 4 {
				x0 := x[j*d : j*d+d]
				x1 := x[(j+1)*d : (j+1)*d+d]
				x2 := x[(j+2)*d : (j+2)*d+d]
				x3 := x[(j+3)*d : (j+3)*d+d]
				for r := 0; r < rows; r++ {
					wr := buf[r*d : r*d+d]
					s0, s1, s2, s3 := dot4(wr, x0, x1, x2, x3)
					i := b0 + r
					dst[j*n+i] = s0
					dst[(j+1)*n+i] = s1
					dst[(j+2)*n+i] = s2
					dst[(j+3)*n+i] = s3
				}
			}
			for ; j < m; j++ {
				xj := x[j*d : j*d+d]
				for r := 0; r < rows; r++ {
					dst[j*n+b0+r] = dot(buf[r*d:r*d+d], xj)
				}
			}
		}
	})
}

// matvec is matmul with m == 1.
func matvec(dst []float32, w *gguf.Tensor, x []float32) {
	matmul(dst, w, x, 1)
}
