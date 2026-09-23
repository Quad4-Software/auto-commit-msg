// SPDX-License-Identifier: 0BSD

//go:build amd64

#include "textflag.h"

DATA m0f<>+0(SB)/4, $0x0f0f0f0f
DATA m30<>+0(SB)/4, $0x30303030
DATA b32<>+0(SB)/4, $0x20202020
GLOBL m0f<>(SB), RODATA, $4
GLOBL m30<>(SB), RODATA, $4
GLOBL b32<>(SB), RODATA, $4

// expand8: convert 8 int32 lanes in src xmm to float32, fma with d/-m,
// store at off(BX). Clobbers Y4, Y5, X6.
#define EXPAND8(src, dv, mv, off) \
	VPMOVZXBD   src, Y4          \
	VCVTDQ2PS   Y4, Y4           \
	VMOVAPS     mv, Y5           \
	VFMADD231PS Y4, dv, Y5       \
	VMOVUPS     Y5, off(BX)

// func q4kExpand64(qs *byte, dst *float32, d1, m1, d2, m2 float32)
// Expands 32 packed bytes into 64 floats:
//   dst[0:32]  = d1 * lo_nibble - m1   (m1 passed negated)
//   dst[32:64] = d2 * hi_nibble - m2   (m2 passed negated)
TEXT ·q4kExpand64(SB), NOSPLIT, $0-32
	MOVQ        qs+0(FP), AX
	MOVQ        dst+8(FP), BX
	VBROADCASTSS d1+16(FP), Y10
	VBROADCASTSS m1+20(FP), Y11
	VBROADCASTSS d2+24(FP), Y12
	VBROADCASTSS m2+28(FP), Y13
	VBROADCASTSS m0f<>+0(SB), Y14
	VMOVDQU     (AX), Y8
	VPAND       Y14, Y8, Y9   // lo nibbles
	VPSRLW      $4, Y8, Y8
	VPAND       Y14, Y8, Y8   // hi nibbles
	// lo half, elems 0-31, scale d1
	EXPAND8(X9, Y10, Y11, 0)
	VPSRLDQ     $8, X9, X6
	EXPAND8(X6, Y10, Y11, 32)
	VEXTRACTI128 $1, Y9, X6
	EXPAND8(X6, Y10, Y11, 64)
	VPSRLDQ     $8, X6, X6
	EXPAND8(X6, Y10, Y11, 96)
	// hi half, elems 32-63, scale d2
	EXPAND8(X8, Y12, Y13, 128)
	VPSRLDQ     $8, X8, X6
	EXPAND8(X6, Y12, Y13, 160)
	VEXTRACTI128 $1, Y8, X6
	EXPAND8(X6, Y12, Y13, 192)
	VPSRLDQ     $8, X6, X6
	EXPAND8(X6, Y12, Y13, 224)
	VZEROUPPER
	RET

// expand8s: like EXPAND8 but result is q - 32 (6-bit signed) then scaled
// by d broadcast in Y15. Sub 32 folded into the int domain via VPSUBB
// before widening; clobbers Y4, Y5.
#define EXPAND8S(src, dv, off) \
	VPMOVSXBD   src, Y4          \
	VCVTDQ2PS   Y4, Y4           \
	VMULPS      dv, Y4, Y5       \
	VMOVUPS     Y5, off(BX)

// func q6kHalf(ql, qh *byte, ds *float32, dst *float32)
// Expands one 128-element half of a Q6_K super-block.
// ds holds the 8 per-16-element scales pre-multiplied by d.
TEXT ·q6kHalf(SB), NOSPLIT, $0-32
	MOVQ         ql+0(FP), AX
	MOVQ         qh+8(FP), CX
	MOVQ         ds+16(FP), DX
	MOVQ         dst+24(FP), BX
	VBROADCASTSS m0f<>+0(SB), Y14
	VBROADCASTSS m30<>+0(SB), Y15
	VBROADCASTSS b32<>+0(SB), Y13
	VMOVDQU      (AX), Y8     // ql[0:32]
	VMOVDQU      32(AX), Y9   // ql[32:64]
	VMOVDQU      (CX), Y10    // qh[0:32]

	// stream 1: elems 0-31, q = (ql&0x0f) | ((qh&3)<<4)
	VPAND  Y14, Y8, Y0
	VPSLLW $4, Y10, Y11
	VPAND  Y15, Y11, Y11
	VPOR   Y11, Y0, Y0
	VPSUBB Y13, Y0, Y0        // int8 q in [-32,31]
	VBROADCASTSS (DX), Y12
	EXPAND8S(X0, Y12, 0)
	VPSRLDQ      $8, X0, X6
	EXPAND8S(X6, Y12, 32)
	VEXTRACTI128 $1, Y0, X6
	VBROADCASTSS 4(DX), Y12
	EXPAND8S(X6, Y12, 64)
	VPSRLDQ      $8, X6, X6
	EXPAND8S(X6, Y12, 96)

	// stream 2: elems 32-63, q = (ql[32+l]&0x0f) | ((qh<<2)&0x30)
	VPAND  Y14, Y9, Y0
	VPSLLW $2, Y10, Y11
	VPAND  Y15, Y11, Y11
	VPOR   Y11, Y0, Y0
	VPSUBB Y13, Y0, Y0
	VBROADCASTSS 8(DX), Y12
	EXPAND8S(X0, Y12, 128)
	VPSRLDQ      $8, X0, X6
	EXPAND8S(X6, Y12, 160)
	VEXTRACTI128 $1, Y0, X6
	VBROADCASTSS 12(DX), Y12
	EXPAND8S(X6, Y12, 192)
	VPSRLDQ      $8, X6, X6
	EXPAND8S(X6, Y12, 224)

	// stream 3: elems 64-95, q = (ql>>4) | (qh&0x30)
	VPSRLW $4, Y8, Y0
	VPAND  Y14, Y0, Y0
	VPAND  Y15, Y10, Y11
	VPOR   Y11, Y0, Y0
	VPSUBB Y13, Y0, Y0
	VBROADCASTSS 16(DX), Y12
	EXPAND8S(X0, Y12, 256)
	VPSRLDQ      $8, X0, X6
	EXPAND8S(X6, Y12, 288)
	VEXTRACTI128 $1, Y0, X6
	VBROADCASTSS 20(DX), Y12
	EXPAND8S(X6, Y12, 320)
	VPSRLDQ      $8, X6, X6
	EXPAND8S(X6, Y12, 352)

	// stream 4: elems 96-127, q = (ql[32+l]>>4) | ((qh>>2)&0x30)
	VPSRLW $4, Y9, Y0
	VPAND  Y14, Y0, Y0
	VPSRLW $2, Y10, Y11
	VPAND  Y15, Y11, Y11
	VPOR   Y11, Y0, Y0
	VPSUBB Y13, Y0, Y0
	VBROADCASTSS 24(DX), Y12
	EXPAND8S(X0, Y12, 384)
	VPSRLDQ      $8, X0, X6
	EXPAND8S(X6, Y12, 416)
	VEXTRACTI128 $1, Y0, X6
	VBROADCASTSS 28(DX), Y12
	EXPAND8S(X6, Y12, 448)
	VPSRLDQ      $8, X6, X6
	EXPAND8S(X6, Y12, 480)
	VZEROUPPER
	RET

// func q80Expand32(qs *byte, dst *float32, d float32)
// Expands one 32-element Q8_0 block: dst[i] = d * int8(qs[i]).
TEXT ·q80Expand32(SB), NOSPLIT, $0-20
	MOVQ         qs+0(FP), AX
	MOVQ         dst+8(FP), BX
	VBROADCASTSS d+16(FP), Y15
	VMOVDQU      (AX), Y8
	EXPAND8S(X8, Y15, 0)
	VPSRLDQ      $8, X8, X6
	EXPAND8S(X6, Y15, 32)
	VEXTRACTI128 $1, Y8, X6
	EXPAND8S(X6, Y15, 64)
	VPSRLDQ      $8, X6, X6
	EXPAND8S(X6, Y15, 96)
	VZEROUPPER
	RET

// func q40Expand32(qs *byte, dst *float32, d, bias float32)
// Expands one 32-element Q4_0 block:
//   dst[0:16]  = d * lo_nibble + bias   (bias = -8*d, passed precomputed)
//   dst[16:32] = d * hi_nibble + bias
TEXT ·q40Expand32(SB), NOSPLIT, $0-24
	MOVQ         qs+0(FP), AX
	MOVQ         dst+8(FP), BX
	VBROADCASTSS d+16(FP), Y10
	VBROADCASTSS bias+20(FP), Y11
	VBROADCASTSS m0f<>+0(SB), Y14
	VMOVDQU      (AX), X8      // 16 packed bytes
	VPAND        X14, X8, X9   // lo nibbles
	VPSRLW       $4, X8, X8
	VPAND        X14, X8, X8   // hi nibbles
	EXPAND8(X9, Y10, Y11, 0)
	VPSRLDQ      $8, X9, X6
	EXPAND8(X6, Y10, Y11, 32)
	EXPAND8(X8, Y10, Y11, 64)
	VPSRLDQ      $8, X8, X6
	EXPAND8(X6, Y10, Y11, 96)
	VZEROUPPER
	RET
