// SPDX-License-Identifier: 0BSD

//go:build !unix

package app

import "os"

func readModel(path string) ([]byte, error) {
	return os.ReadFile(path)
}
