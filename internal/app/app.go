// SPDX-License-Identifier: 0BSD

// Package app wires the diff collection, prompt, model loading and
// generation together.
package app

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"quad4.io/auto-commit-msg/internal/bpe"
	"quad4.io/auto-commit-msg/internal/gguf"
	"quad4.io/auto-commit-msg/internal/infer"
)

// DefaultModelURL is base Qwen3-0.6B quantized to Q4_K_M. Any GGUF with a
// qwen2 or qwen3 architecture works; pass -model or set AUTO_COMMIT_MSG_MODEL.
const DefaultModelURL = "https://huggingface.co/bartowski/Qwen_Qwen3-0.6B-GGUF/resolve/main/Qwen_Qwen3-0.6B-Q4_K_M.gguf"

// Embedded is set by the bundle build tag (see embed_bundle.go).
var Embedded []byte

type Options struct {
	ModelPath     string
	Conventional  bool
	All           bool // diff HEAD instead of just staged
	MaxDiffTokens int
	MaxGenTokens  int
	Context       int
	Temp          float64
	TopP          float64
	Verbose       bool
}

func (o *Options) defaults() {
	if o.MaxDiffTokens <= 0 {
		o.MaxDiffTokens = 1024
	}
	if o.MaxGenTokens <= 0 {
		o.MaxGenTokens = 64
	}
	if o.Context <= 0 {
		o.Context = 2048
	}
	if o.TopP <= 0 {
		o.TopP = 0.9
	}
}

func modelDir() string {
	if d := os.Getenv("XDG_DATA_HOME"); d != "" {
		return filepath.Join(d, "auto-commit-msg")
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "auto-commit-msg")
}

func modelPath(flag string) string {
	if flag != "" {
		return flag
	}
	if p := os.Getenv("AUTO_COMMIT_MSG_MODEL"); p != "" {
		return p
	}
	return filepath.Join(modelDir(), "model.gguf")
}

// loadModel resolves the weights: flag/env path, then embedded blob, then
// the XDG data dir.
func loadModel(o *Options) (*gguf.File, error) {
	path := modelPath(o.ModelPath)
	if data, err := readModel(path); err == nil {
		return gguf.Parse(data)
	}
	if len(Embedded) > 0 {
		return gguf.Parse(Embedded)
	}
	if o.ModelPath != "" {
		return nil, fmt.Errorf("cannot read model %q", o.ModelPath)
	}
	return nil, fmt.Errorf("no model found; run `auto-commit-msg setup` or build with `make fat`")
}

// diff returns the staged diff, or `git diff HEAD` when o.All.
// AUTO_COMMIT_MSG_DIFF overrides git and reads the diff from a file.
func diff(o *Options) (string, error) {
	if p := os.Getenv("AUTO_COMMIT_MSG_DIFF"); p != "" {
		b, err := os.ReadFile(p)
		if err != nil {
			return "", err
		}
		if strings.TrimSpace(string(b)) == "" {
			return "", fmt.Errorf("no changes to describe")
		}
		return string(b), nil
	}
	args := []string{"diff", "--staged"}
	if o.All {
		args = []string{"diff", "HEAD"}
	}
	cmd := exec.Command("git", args...)
	out, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("git %s: %w", strings.Join(args, " "), err)
	}
	if strings.TrimSpace(string(out)) == "" {
		return "", fmt.Errorf("no changes to describe (stage with git add, or pass -all)")
	}
	return string(out), nil
}

func systemPrompt(conventional bool) string {
	if conventional {
		return "You write git commit messages in Conventional Commits format: " +
			"type(scope): subject. Types: feat, fix, docs, style, refactor, perf, " +
			"test, build, ci, chore, revert. Subject in imperative mood, under " +
			"72 characters. Output only the message.\n\n" +
			"Examples:\n" +
			"feat(parser): handle empty input\n" +
			"fix(api): return 404 for missing user\n" +
			"chore(deps): bump go to 1.27"
	}
	return "You write git commit messages. Given a diff, reply with a single " +
		"concise subject line in imperative mood, under 72 characters, " +
		"describing the change. Output only the message, no quotes or explanation."
}

// committedSys is the exact system instruction the Committed fine-tune was
// trained on. Deviating from it degrades output, so it is used verbatim when
// a committed-* GGUF is detected (general.basename contains "committed").
const committedSys = `You write a single Conventional Commits message describing a git diff.
Pick the type that best matches what changed:
feat - adds a capability; fix - corrects a bug; docs - documentation only; style - formatting with no change in logic; refactor - restructures code without changing behavior; perf - improves performance; test - adds or fixes tests; build - build system or dependencies; ci - CI configuration; chore - maintenance touching neither source nor tests.
Add a scope in parentheses only when a single file or area clearly owns the change; if the change is spread out or the owner is unclear, omit it.
Write the description so that:
- It reads correctly after "If applied, this commit will..." - imperative verb first ("add", never "adds" or "added").
- It states only what the diff shows. You can see what changed, not why, so never invent a reason, motivation, or outcome the diff doesn't contain; when unsure, say less rather than guess.
- It names the most significant change when the diff touches several things.
- It is specific: name the real function, file, flag, or endpoint, and skip filler verbs ("update", "change") and vague objects ("code", "stuff") when something precise fits.
`

var conventionalRe = regexp.MustCompile(`^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)(\([a-z0-9._/-]+\))?!?: .+`)

func isCommitted(f *gguf.File) bool {
	return strings.Contains(strings.ToLower(f.Str("general.basename")), "committed")
}

// buildPrompt wraps the system prompt and diff in the chat template the
// model was trained with, detected from its vocab. committed-* models get
// their exact training prompt verbatim.
func buildPrompt(f *gguf.File, tok *bpe.Tokenizer, arch, diffText string, conventional bool) string {
	if strings.Contains(strings.ToLower(f.Str("general.basename")), "committed") {
		return "<|im_start|>system\n" + committedSys + "<|im_end|>\n" +
			"<|im_start|>user\nDiff:\n" + diffText + "\n\n/no_think<|im_end|>\n" +
			"<|im_start|>assistant\n<think>\n\n</think>\n\n"
	}
	sys := systemPrompt(conventional)
	if _, ok := tok.ID("<|start_header_id|>"); ok {
		// llama3 header format
		return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
			sys + "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" +
			diffText + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
	}
	if _, ok := tok.ID("<|im_start|>"); ok {
		// ChatML; the empty think block only exists on qwen3
		think := ""
		if arch == "qwen3" {
			think = "<think>\n\n</think>\n\n"
		}
		return "<|im_start|>system\n" + sys + "<|im_end|>\n" +
			"<|im_start|>user\n" + diffText + "<|im_end|>\n" +
			"<|im_start|>assistant\n" + think
	}
	// no chat template tokens: plain fallback
	return "System: " + sys + "\n\nUser: " + diffText + "\n\nAssistant: "
}

// cleanup extracts a single subject line from raw model output.
func cleanup(s string) string {
	// drop <think>...</think> blocks if the model emitted them anyway
	if i := strings.Index(s, "</think>"); i >= 0 {
		s = s[i+len("</think>"):]
	}
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(line)
		// small models often echo the command itself
		for _, pre := range []string{"git commit", "git ci", "-m"} {
			line = strings.TrimPrefix(line, pre)
			line = strings.TrimSpace(line)
		}
		// bare "commit" only when followed by a flag or quote
		if rest, ok := strings.CutPrefix(line, "commit"); ok &&
			(strings.HasPrefix(rest, " ") && strings.ContainsAny(rest[1:], "\"'-")) {
			line = strings.TrimSpace(rest)
		}
		line = strings.TrimPrefix(line, "Commit message:")
		line = strings.TrimSpace(line)
		line = strings.Trim(line, "\"'` ")
		if line != "" {
			return line
		}
	}
	return ""
}

// peakRSS reports the process high-water RSS in MB, or 0 if unknown.
func peakRSS() float64 {
	b, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(b), "\n") {
		if strings.HasPrefix(line, "VmHWM:") {
			var kb float64
			fmt.Sscanf(line, "VmHWM: %f kB", &kb)
			return kb / 1024
		}
	}
	return 0
}

// Generate produces the commit message.
func Generate(o *Options) (string, error) {
	o.defaults()
	t0 := time.Now()
	diffText, err := diff(o)
	if err != nil {
		return "", err
	}
	f, err := loadModel(o)
	if err != nil {
		return "", err
	}
	tokens := f.Strs("tokenizer.ggml.tokens")
	// token_type: 3 = control, 4 = user-defined; both matched literally
	var specials []int
	for i, tt := range f.Ints("tokenizer.ggml.token_type") {
		if tt == 3 || tt == 4 {
			specials = append(specials, i)
		}
	}
	tok := bpe.New(tokens, f.Strs("tokenizer.ggml.merges"),
		int(f.U64("tokenizer.ggml.bos_token_id", 0)),
		int(f.U64("tokenizer.ggml.eos_token_id", 0)), specials,
		f.Str("tokenizer.ggml.pre"))
	m, err := infer.Load(f, o.Context)
	if err != nil {
		return "", err
	}
	loadTime := time.Since(t0)

	diffIDs := tok.Encode(diffText)
	diffText = compressDiff(diffText, o.MaxDiffTokens, tok)

	prompt := buildPrompt(f, tok, m.Cfg.Arch, diffText, o.Conventional)
	ids := tok.Encode(prompt)
	if f.U64("tokenizer.ggml.add_bos_token", 0) == 1 &&
		(len(ids) == 0 || ids[0] != m.Cfg.BOS) {
		ids = append([]int{m.Cfg.BOS}, ids...)
	}
	if o.Verbose {
		fmt.Fprintf(os.Stderr, "prompt: %d tokens (diff %d), ctx %d, load %v\n",
			len(ids), len(diffIDs), o.Context, loadTime.Round(time.Millisecond))
	}

	var gen []int
	for attempt := 0; attempt < 2; attempt++ {
		temp := o.Temp
		if attempt > 0 {
			m.Reset()
			temp = 0.5 // retry with mild sampling on format failure
		}
		s := &infer.Sampler{Temp: temp, TopP: o.TopP, Rng: newRand()}
		var gerr error
		gen, gerr = m.Generate(ids, s, o.MaxGenTokens, nil)
		if gerr != nil {
			return "", gerr
		}
		if o.Verbose {
			st := m.Stats
			fmt.Fprintf(os.Stderr, "prefill: %d tok in %v (%.0f tok/s), decode: %d tok in %v (%.1f tok/s), peak RSS %.0f MB\n",
				st.PrefillToks, st.Prefill.Round(time.Millisecond), float64(st.PrefillToks)/st.Prefill.Seconds(),
				st.DecodeToks, st.Decode.Round(time.Millisecond), float64(st.DecodeToks)/st.Decode.Seconds(),
				peakRSS())
			fmt.Fprintf(os.Stderr, "raw: %q\n", tok.Decode(gen))
		}
		msg := cleanup(tok.Decode(gen))
		if msg == "" {
			continue
		}
		if (o.Conventional || isCommitted(f)) && !conventionalRe.MatchString(msg) {
			continue
		}
		return msg, nil
	}
	msg := cleanup(tok.Decode(gen))
	if msg == "" {
		return "", fmt.Errorf("model produced no message")
	}
	return "", fmt.Errorf("model output does not match conventional format: %q", msg)
}

// Setup downloads the default model into the XDG data dir.
func Setup(url string) error {
	dir := modelDir()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	dst := filepath.Join(dir, "model.gguf")
	if _, err := os.Stat(dst); err == nil {
		fmt.Fprintf(os.Stderr, "model already at %s\n", dst)
		return nil
	}
	fmt.Fprintf(os.Stderr, "downloading %s\n", url)
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		return fmt.Errorf("download failed: %s", resp.Status)
	}
	tmp := dst + ".part"
	out, err := os.Create(tmp)
	if err != nil {
		return err
	}
	n, err := io.Copy(out, resp.Body)
	out.Close()
	if err != nil {
		os.Remove(tmp)
		return err
	}
	if err := os.Rename(tmp, dst); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "wrote %s (%.0f MB)\n", dst, float64(n)/1e6)
	return nil
}

const hook = `#!/bin/sh
# auto-commit-msg prepare-commit-msg hook
# fills the message for plain commits; -m, --amend, merge, squash untouched
case "$2" in
""|template)
	msg=$(auto-commit-msg 2>/dev/null)
	if [ -n "$msg" ]; then
		{ printf '%s\n\n' "$msg"; cat "$1"; } > "$1.tmp" && mv "$1.tmp" "$1"
	fi
	;;
esac
`

// InstallHook writes .git/hooks/prepare-commit-msg.
func InstallHook() error {
	out, err := exec.Command("git", "rev-parse", "--git-dir").Output()
	if err != nil {
		return fmt.Errorf("not a git repository")
	}
	dir := filepath.Join(strings.TrimSpace(string(out)), "hooks")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return err
	}
	p := filepath.Join(dir, "prepare-commit-msg")
	if _, err := os.Stat(p); err == nil {
		return fmt.Errorf("%s already exists", p)
	}
	if err := os.WriteFile(p, []byte(hook), 0o755); err != nil {
		return err
	}
	fmt.Fprintf(os.Stderr, "installed %s\n", p)
	return nil
}
