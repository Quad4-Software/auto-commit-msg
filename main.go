// SPDX-License-Identifier: 0BSD

// auto-commit-msg generates a git commit message from the staged diff
// using a local GGUF model. Pure Go, no cgo, fully offline.
package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"

	"quad4.io/auto-commit-msg/internal/app"
)

func main() {
	var o app.Options
	var commit, edit bool
	fs := flag.NewFlagSet("auto-commit-msg", flag.ExitOnError)
	fs.StringVar(&o.ModelPath, "model", "", "path to a GGUF model (env: AUTO_COMMIT_MSG_MODEL)")
	fs.BoolVar(&commit, "commit", false, "commit with the generated message")
	fs.BoolVar(&commit, "c", false, "commit with the generated message")
	fs.BoolVar(&edit, "edit", false, "commit and open the editor on the message")
	fs.BoolVar(&edit, "e", false, "commit and open the editor on the message")
	fs.BoolVar(&o.Conventional, "conventional", false, "enforce type(scope): subject format")
	fs.BoolVar(&o.All, "all", false, "describe all changes vs HEAD, not just staged")
	fs.IntVar(&o.MaxDiffTokens, "max-diff-tokens", 1024, "truncate the diff to this many tokens")
	fs.IntVar(&o.MaxGenTokens, "max-tokens", 64, "max generated tokens")
	fs.IntVar(&o.Context, "ctx", 2048, "context window")
	fs.Float64Var(&o.Temp, "temp", 0, "sampling temperature (0 = greedy)")
	fs.Float64Var(&o.TopP, "top-p", 0.9, "nucleus sampling p")
	fs.BoolVar(&o.Verbose, "v", false, "verbose output to stderr")
	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "usage: auto-commit-msg [flags] | setup | install-hook\n")
		fs.PrintDefaults()
	}
	fs.Parse(os.Args[1:])

	switch fs.Arg(0) {
	case "setup":
		url := app.DefaultModelURL
		if fs.Arg(1) != "" {
			url = fs.Arg(1)
		}
		if err := app.Setup(url); err != nil {
			fatal(err)
		}
		return
	case "install-hook":
		if err := app.InstallHook(); err != nil {
			fatal(err)
		}
		return
	}

	msg, err := app.Generate(&o)
	if err != nil {
		fatal(err)
	}
	switch {
	case commit, edit:
		args := []string{"commit", "-m", msg}
		if edit {
			args = []string{"commit", "-e", "-m", msg}
		}
		fmt.Fprintln(os.Stderr, msg)
		cmd := exec.Command("git", args...)
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
		cmd.Stdin = os.Stdin
		if err := cmd.Run(); err != nil {
			fatal(err)
		}
	default:
		fmt.Println(msg)
	}
}

func fatal(err error) {
	fmt.Fprintln(os.Stderr, "auto-commit-msg:", err)
	os.Exit(1)
}
