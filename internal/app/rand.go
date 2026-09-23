// SPDX-License-Identifier: 0BSD

package app

import (
	"math/rand/v2"
	"time"
)

func newRand() *rand.Rand {
	now := uint64(time.Now().UnixNano())
	return rand.New(rand.NewPCG(now, now^0x9e3779b97f4a7c15))
}
