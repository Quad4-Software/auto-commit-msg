// SPDX-License-Identifier: 0BSD

//go:build amd64

package infer

import "quad4.io/auto-commit-msg/internal/gguf"

// asm kernels; see dequant_amd64.s
func q4kExpand64(qs *byte, dst *float32, d1, m1, d2, m2 float32)
func q6kHalf(ql, qh *byte, ds *float32, dst *float32)
func q80Expand32(qs *byte, dst *float32, d float32)
func q40Expand32(qs *byte, dst *float32, d, bias float32)

// dequantRowFast dequantizes row r of tensor t into dst using AVX2 kernels.
func dequantRowFast(t *gguf.Tensor, r int, dst []float32) {
	row := t.Data[r*t.RowBytes():]
	switch t.Type {
	case gguf.TypeQ4_K:
		dequantQ4KFast(row, dst)
	case gguf.TypeQ6_K:
		dequantQ6KFast(row, dst)
	case gguf.TypeQ8_0:
		dequantQ80Fast(row, dst)
	case gguf.TypeQ4_0:
		dequantQ40Fast(row, dst)
	case gguf.TypeF32:
		dequantRowSlow(t, r, dst)
	case gguf.TypeF16:
		dequantRowSlow(t, r, dst)
	default:
		dequantRowSlow(t, r, dst)
	}
}

func dequantQ4KFast(row []byte, dst []float32) {
	for nb := 0; nb*256 < len(dst); nb++ {
		blk := row[nb*144:]
		d := f16(blk)
		dmin := f16(blk[2:])
		sc := blk[4:16]
		qs := blk[16:]
		y := dst[nb*256:]
		for g := 0; g < 4; g++ {
			s1, m1 := scaleMinK4(2*g, sc)
			s2, m2 := scaleMinK4(2*g+1, sc)
			q4kExpand64(&qs[g*32], &y[g*64],
				d*float32(s1), -dmin*float32(m1),
				d*float32(s2), -dmin*float32(m2))
		}
	}
}

func dequantQ6KFast(row []byte, dst []float32) {
	for nb := 0; nb*256 < len(dst); nb++ {
		blk := row[nb*210:]
		ql := blk[:128]
		qh := blk[128:192]
		sc := blk[192:208]
		d := f16(blk[208:])
		y := dst[nb*256:]
		for i := 0; i < 2; i++ {
			var ds [8]float32
			for j := 0; j < 8; j++ {
				ds[j] = d * float32(int8(sc[8*i+j]))
			}
			q6kHalf(&ql[64*i], &qh[32*i], &ds[0], &y[128*i])
		}
	}
}

func dequantQ80Fast(row []byte, dst []float32) {
	for b := 0; b*32 < len(dst); b++ {
		q80Expand32(&row[b*34+2], &dst[b*32], f16(row[b*34:]))
	}
}

func dequantQ40Fast(row []byte, dst []float32) {
	for b := 0; b*32 < len(dst); b++ {
		d := f16(row[b*18:])
		q40Expand32(&row[b*18+2], &dst[b*32], d, -8*d)
	}
}
