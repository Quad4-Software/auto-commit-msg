// SPDX-License-Identifier: 0BSD

//go:build bundle

package main

import (
	_ "embed"

	"quad4.io/auto-commit-msg/internal/app"
)

//go:embed model.gguf
var modelGGUF []byte

func init() { app.Embedded = modelGGUF }
