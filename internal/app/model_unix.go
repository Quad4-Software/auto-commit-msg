// SPDX-License-Identifier: 0BSD

//go:build unix

package app

import (
	"os"
	"syscall"
)

// readModel maps the model file read-only. The mapping is file-backed so
// weight pages stay out of the Go heap and are reclaimable by the OS.
// It is never unmapped; the process exits soon after inference anyway.
func readModel(path string) ([]byte, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()),
		syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		return nil, err
	}
	// ask the kernel to start paging the weights in while we parse
	// metadata and build the tokenizer
	_ = syscall.Madvise(b, syscall.MADV_WILLNEED)
	return b, nil
}
