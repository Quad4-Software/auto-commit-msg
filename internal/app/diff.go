// SPDX-License-Identifier: 0BSD

package app

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"

	"quad4.io/auto-commit-msg/internal/bpe"
)

// noiseFiles match diffs that carry no intent for a commit message:
// lockfiles, generated code, minified assets, vendored trees.
var noiseFiles = regexp.MustCompile(`(?i)(^|/)` +
	`(go\.sum|go\.work\.sum|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|` +
	`composer\.lock|cargo\.lock|gemfile\.lock|poetry\.lock|flake\.lock|` +
	`package\.lock|bun\.lock(b)?|deno\.lock|pubspec\.lock|mix\.lock|` +
	`.*\.min\.(js|css)|.*\.map|.*\.pb\.go|.*_generated\..*|.*\.gen\..*)$`)

// fileDiff is one file's section of a unified diff.
type fileDiff struct {
	path string
	head string // header lines: diff --git, ---/+++ etc
	body []string
}

// splitDiff splits a unified diff into per-file sections.
func splitDiff(d string) []fileDiff {
	var out []fileDiff
	var cur *fileDiff
	for _, ln := range strings.Split(d, "\n") {
		if strings.HasPrefix(ln, "diff --git ") {
			out = append(out, fileDiff{})
			cur = &out[len(out)-1]
			cur.path = diffPath(ln)
			cur.head = ln + "\n"
			continue
		}
		if cur == nil {
			continue
		}
		switch {
		case strings.HasPrefix(ln, "index "):
			// blob hashes carry no meaning
		case strings.HasPrefix(ln, "Binary files"):
			cur.head += ln + "\n"
		case strings.HasPrefix(ln, "new file"), strings.HasPrefix(ln, "deleted file"),
			strings.HasPrefix(ln, "old mode"), strings.HasPrefix(ln, "new mode"),
			strings.HasPrefix(ln, "similarity"), strings.HasPrefix(ln, "rename"),
			strings.HasPrefix(ln, "copy "), strings.HasPrefix(ln, "---"),
			strings.HasPrefix(ln, "+++"):
			cur.head += ln + "\n"
		default:
			cur.body = append(cur.body, ln)
		}
	}
	return out
}

func diffPath(ln string) string {
	// "diff --git a/path b/path"; use the b/ side
	if i := strings.LastIndex(ln, " b/"); i >= 0 {
		return ln[i+3:]
	}
	return ln
}

// maxHunkLines caps the +/- body lines kept per file.
const maxHunkLines = 100

// renderDiff rebuilds a file section, truncating the body.
func renderDiff(f fileDiff) (string, int) {
	if noiseFiles.MatchString(filepath.ToSlash(f.path)) {
		return f.head + "(generated file, contents omitted)\n", len(f.body)
	}
	var b strings.Builder
	b.WriteString(f.head)
	body := f.body
	cut := 0
	if len(body) > maxHunkLines {
		body = body[:maxHunkLines]
		cut = len(f.body) - maxHunkLines
	}
	b.WriteString(strings.Join(body, "\n"))
	if cut > 0 {
		fmt.Fprintf(&b, "... (%d more lines)\n", cut)
	}
	return b.String(), len(f.body)
}

// cleanDiff elides noise file bodies (lockfiles, generated code) but keeps
// their headers so the model still sees that the file changed. Always applied,
// even when the diff fits the token budget.
func cleanDiff(d string) string {
	files := splitDiff(d)
	if len(files) == 0 {
		return d
	}
	var b strings.Builder
	changed := false
	for _, f := range files {
		if noiseFiles.MatchString(filepath.ToSlash(f.path)) {
			b.WriteString(f.head + "(generated file, contents omitted)\n\n")
			changed = true
			continue
		}
		b.WriteString(f.head)
		b.WriteString(strings.Join(f.body, "\n"))
		b.WriteString("\n")
	}
	if !changed {
		return d
	}
	return b.String()
}

// compressDiff reduces a diff to fit a token budget. Files keep their
// headers and hunk markers; oversized sections are truncated per file
// and fully dropped files are listed by name at the end.
func compressDiff(d string, budget int, tok *bpe.Tokenizer) string {
	d = cleanDiff(d)
	if tok == nil || len(tok.Encode(d)) <= budget {
		return d
	}
	files := splitDiff(d)
	var b strings.Builder
	var dropped []string
	for _, f := range files {
		sec, _ := renderDiff(f)
		n := len(tok.Encode(sec))
		if n > budget {
			dropped = append(dropped, f.path)
			continue
		}
		b.WriteString(sec)
		b.WriteString("\n")
		budget -= n
	}
	out := b.String()
	if len(dropped) > 0 {
		names := dropped
		if len(names) > 12 {
			names = append(names[:12], fmt.Sprintf("and %d more", len(dropped)-12))
		}
		out += fmt.Sprintf("[omitted %d files: %s]\n", len(dropped), strings.Join(names, ", "))
	}
	if strings.TrimSpace(out) == "" {
		// budget too small even for headers; fall back to hard cut
		ids := tok.Encode(d)
		return tok.Decode(ids[:min(budget, len(ids))]) + "\n[diff truncated]"
	}
	return out + "[diff truncated]"
}
