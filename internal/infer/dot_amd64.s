// SPDX-License-Identifier: 0BSD

//go:build amd64

#include "textflag.h"

// func cpuHasAVX2() bool
TEXT ·cpuHasAVX2(SB), NOSPLIT, $0-1
	MOVL $1, AX
	CPUID
	MOVL CX, BX
	ANDL $(1<<27 | 1<<28 | 1<<12), BX // OSXSAVE | AVX | FMA
	CMPL BX, $(1<<27 | 1<<28 | 1<<12)
	JNE  no
	MOVL $0, CX
	BYTE $0x0F; BYTE $0x01; BYTE $0xD0  // XGETBV -> EDX:EAX
	ANDL $6, AX                        // XMM and YMM state enabled
	CMPL AX, $6
	JNE  no
	MOVL $7, AX
	MOVL $0, CX
	CPUID
	ANDL $(1<<5), BX                   // AVX2
	JZ   no
	MOVB $1, ret+0(FP)
	RET
no:
	MOVB $0, ret+0(FP)
	RET

// func dotAVX2(a, b *float32, n int) float32
// n must be a multiple of 8; the caller handles the tail.
TEXT ·dotAVX2(SB), NOSPLIT, $0-28
	MOVQ   a+0(FP), AX
	MOVQ   b+8(FP), BX
	MOVQ   n+16(FP), CX
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	MOVQ   CX, DX
	SHRQ   $5, DX
	JZ     tail8
loop32:
	VMOVUPS      (AX), Y4
	VMOVUPS    32(AX), Y5
	VMOVUPS    64(AX), Y6
	VMOVUPS    96(AX), Y7
	VFMADD231PS   (BX), Y4, Y0
	VFMADD231PS 32(BX), Y5, Y1
	VFMADD231PS 64(BX), Y6, Y2
	VFMADD231PS 96(BX), Y7, Y3
	ADDQ $128, AX
	ADDQ $128, BX
	DECQ DX
	JNZ  loop32
tail8:
	ANDL $31, CX
	SHRQ $3, CX
	JZ   hsum
loop8:
	VMOVUPS      (AX), Y4
	VFMADD231PS (BX), Y4, Y0
	ADDQ $32, AX
	ADDQ $32, BX
	DECQ CX
	JNZ  loop8
hsum:
	VADDPS        Y0, Y1, Y0
	VADDPS        Y2, Y3, Y2
	VADDPS        Y0, Y2, Y0
	VEXTRACTF128 $1, Y0, X1
	VADDPS        X0, X1, X0
	VHADDPS       X0, X0, X0
	VHADDPS       X0, X0, X0
	VMOVSS        X0, ret+24(FP)
	VZEROUPPER
	RET

// func dot4AVX2(w, a, b, c, d *float32, n int) (s0, s1, s2, s3 float32)
// Computes w.a, w.b, w.c, w.d loading w once per vector. n % 8 == 0.
TEXT ·dot4AVX2(SB), NOSPLIT, $0-64
	MOVQ   w+0(FP), AX
	MOVQ   a+8(FP), BX
	MOVQ   b+16(FP), CX
	MOVQ   c+24(FP), DX
	MOVQ   d+32(FP), SI
	MOVQ   n+40(FP), DI
	VXORPS Y0, Y0, Y0
	VXORPS Y1, Y1, Y1
	VXORPS Y2, Y2, Y2
	VXORPS Y3, Y3, Y3
	VXORPS Y4, Y4, Y4
	VXORPS Y5, Y5, Y5
	VXORPS Y6, Y6, Y6
	VXORPS Y7, Y7, Y7
	MOVQ   DI, R8
	SHRQ   $4, R8
	JZ     tail8
loop16:
	VMOVUPS      (AX), Y8
	VFMADD231PS  (BX), Y8, Y0
	VFMADD231PS  (CX), Y8, Y1
	VFMADD231PS  (DX), Y8, Y2
	VFMADD231PS  (SI), Y8, Y3
	VMOVUPS    32(AX), Y9
	VFMADD231PS 32(BX), Y9, Y4
	VFMADD231PS 32(CX), Y9, Y5
	VFMADD231PS 32(DX), Y9, Y6
	VFMADD231PS 32(SI), Y9, Y7
	ADDQ $64, AX
	ADDQ $64, BX
	ADDQ $64, CX
	ADDQ $64, DX
	ADDQ $64, SI
	DECQ R8
	JNZ  loop16
tail8:
	ANDL $15, DI
	SHRQ $3, DI
	JZ   hsum
loop8:
	VMOVUPS      (AX), Y8
	VFMADD231PS  (BX), Y8, Y0
	VFMADD231PS  (CX), Y8, Y1
	VFMADD231PS  (DX), Y8, Y2
	VFMADD231PS  (SI), Y8, Y3
	ADDQ $32, AX
	ADDQ $32, BX
	ADDQ $32, CX
	ADDQ $32, DX
	ADDQ $32, SI
	DECQ DI
	JNZ  loop8
hsum:
	VADDPS Y4, Y0, Y0
	VADDPS Y5, Y1, Y1
	VADDPS Y6, Y2, Y2
	VADDPS Y7, Y3, Y3
	VEXTRACTF128 $1, Y0, X4
	VADDPS        X0, X4, X0
	VHADDPS       X0, X0, X0
	VHADDPS       X0, X0, X0
	VMOVSS        X0, s0+48(FP)
	VEXTRACTF128 $1, Y1, X4
	VADDPS        X1, X4, X1
	VHADDPS       X1, X1, X1
	VHADDPS       X1, X1, X1
	VMOVSS        X1, s1+52(FP)
	VEXTRACTF128 $1, Y2, X4
	VADDPS        X2, X4, X2
	VHADDPS       X2, X2, X2
	VHADDPS       X2, X2, X2
	VMOVSS        X2, s2+56(FP)
	VEXTRACTF128 $1, Y3, X4
	VADDPS        X3, X4, X3
	VHADDPS       X3, X3, X3
	VHADDPS       X3, X3, X3
	VMOVSS        X3, s3+60(FP)
	VZEROUPPER
	RET

// func axpyAVX2(dst *float32, w float32, src *float32, n int)
// dst[i] += w * src[i]. n must be a multiple of 8.
TEXT ·axpyAVX2(SB), NOSPLIT, $0-32
	MOVQ         dst+0(FP), AX
	VBROADCASTSS w+8(FP), Y1
	MOVQ         src+16(FP), BX
	MOVQ         n+24(FP), CX
	SHRQ         $3, CX
loop:
	VMOVUPS      (BX), Y2
	VMOVUPS      (AX), Y3
	VFMADD231PS  Y1, Y2, Y3
	VMOVUPS      Y3, (AX)
	ADDQ         $32, AX
	ADDQ         $32, BX
	DECQ         CX
	JNZ          loop
	VZEROUPPER
	RET
