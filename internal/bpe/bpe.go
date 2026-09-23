// SPDX-License-Identifier: 0BSD

// Package bpe implements the byte-level BPE tokenizer used by Qwen models,
// matching the llama.cpp "qwen2" pre-tokenizer pattern:
//
//	(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])
//	| [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N} | ?[^\s\p{L}\p{N}]+[\r\n]*
//	| \s*[\r\n]+ | \s+(?!\S) | \s+
package bpe

import (
	"fmt"
	"strings"
	"unicode"
)

type Tokenizer struct {
	vocab    map[string]int
	tokens   []string
	merges   map[[2]string]int
	specials []special // control/user tokens matched literally in input
	ByteTok  [256]rune // byte -> printable rune
	runeB    map[rune]byte
	BOS      int
	EOS      int
	pre      int // pre-tokenizer variant
}

// Pre-tokenizer regex families, matching llama.cpp tokenizer.ggml.pre values.
const (
	preQwen2  = iota // qwen2: \p{N} single digits
	preLlama3        // llama3/smollm: \p{N}{1,3}, CJK chars solo
	preGPT2          // classic gpt2: space-prefixed classes
)

func preByName(name string) int {
	switch name {
	case "qwen2":
		return preQwen2
	case "llama-bpe", "llama3", "llama-v3", "smollm":
		return preLlama3
	default:
		return preGPT2
	}
}

type special struct {
	s  string
	id int
}

// New builds a tokenizer from GGUF metadata arrays. specialIDs lists token
// ids (typically control and user-defined token types) that are matched
// literally in the input text; nil means none. pre is the value of
// tokenizer.ggml.pre ("" selects the generic gpt2 splitter).
func New(tokens, merges []string, bos, eos int, specialIDs []int, pre string) *Tokenizer {
	t := &Tokenizer{
		pre:    preByName(pre),
		vocab:  make(map[string]int, len(tokens)),
		tokens: tokens,
		merges: make(map[[2]string]int, len(merges)),
		runeB:  map[rune]byte{},
		BOS:    bos,
		EOS:    eos,
	}
	for i, s := range tokens {
		t.vocab[s] = i
	}
	for i, m := range merges {
		a, b, ok := strings.Cut(m, " ")
		if !ok {
			continue
		}
		t.merges[[2]string{a, b}] = i
	}
	for _, id := range specialIDs {
		if id >= 0 && id < len(tokens) {
			t.specials = append(t.specials, special{tokens[id], id})
		}
	}
	// longest first, so <|im_start|> beats <|im_... prefixes
	for i := range t.specials {
		for j := i + 1; j < len(t.specials); j++ {
			if len(t.specials[j].s) > len(t.specials[i].s) {
				t.specials[i], t.specials[j] = t.specials[j], t.specials[i]
			}
		}
	}
	// GPT-2 byte table: printable bytes keep their code point, the rest are
	// remapped to 256+n in order.
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xa1 && b <= 0xac) || (b >= 0xae) {
			t.ByteTok[b] = rune(b)
		} else {
			t.ByteTok[b] = rune(256 + n)
			n++
		}
		t.runeB[t.ByteTok[b]] = byte(b)
	}
	return t
}

func (t *Tokenizer) ID(tok string) (int, bool) {
	id, ok := t.vocab[tok]
	return id, ok
}

func (t *Tokenizer) Text(id int) string {
	if id < 0 || id >= len(t.tokens) {
		return ""
	}
	return t.tokens[id]
}

// encodeRune maps one input byte (as part of UTF-8 output) to its table rune.
func (t *Tokenizer) encodeBytes(s string) string {
	var b strings.Builder
	b.Grow(len(s) * 2)
	for i := 0; i < len(s); i++ {
		b.WriteRune(t.ByteTok[s[i]])
	}
	return b.String()
}

func isLetter(r rune) bool { return unicode.IsLetter(r) }
func isDigit(r rune) bool  { return unicode.IsNumber(r) }
func isSpace(r rune) bool  { return unicode.IsSpace(r) }
func isPunct(r rune) bool  { return !isSpace(r) && !isLetter(r) && !isDigit(r) }
func isContractionLead(r rune) bool {
	return !isLetter(r) && !isDigit(r) && r != '\r' && r != '\n'
}

// split pre-tokenizes s into pieces following the selected regex family.
func (t *Tokenizer) split(s string) []string {
	rs := []rune(s)
	var out []string
	i := 0
	for i < len(rs) {
		size := t.matchPiece(rs[i:])
		if size == 0 {
			size = 1
		}
		out = append(out, string(rs[i:i+size]))
		i += size
	}
	return out
}

func isCJK(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r)
}

// matchPiece returns the length of the first regex alternative matching at
// position 0 of rs.
func (t *Tokenizer) matchPiece(rs []rune) int {
	r0 := rs[0]
	// 's 't 're 've 'm 'll 'd (case-insensitive)
	if r0 == '\'' && len(rs) > 1 {
		s := strings.ToLower(string(rs[:min(3, len(rs))]))
		for _, c := range []string{"'re", "'ve", "'ll"} {
			if strings.HasPrefix(s, c) {
				return 3
			}
		}
		switch rs[1] {
		case 's', 'S', 't', 'T', 'm', 'M', 'd', 'D':
			return 2
		}
	}
	// smollm splits CJK characters into single-char pieces
	if t.pre == preLlama3 && isCJK(r0) {
		return 1
	}
	// letters: [^\r\n\p{L}\p{N}]?\p{L}+ (qwen2/llama3) or " ?\p{L}+" (gpt2)
	j := 0
	if t.pre == preGPT2 {
		if r0 == ' ' && len(rs) > 1 && isLetter(rs[1]) {
			j = 1
		}
	} else if !isLetter(r0) && isContractionLead(r0) {
		j = 1
	}
	k := j
	for k < len(rs) && isLetter(rs[k]) {
		k++
	}
	if k > j {
		return k
	}
	// digits: \p{N} (qwen2), \p{N}{1,3} (llama3), " ?\p{N}+" (gpt2)
	if t.pre == preGPT2 {
		j = 0
		if r0 == ' ' && len(rs) > 1 && isDigit(rs[1]) {
			j = 1
		}
		k = j
		for k < len(rs) && isDigit(rs[k]) {
			k++
		}
		if k > j {
			return k
		}
	} else if isDigit(r0) {
		if t.pre == preLlama3 {
			n := 1
			for n < 3 && n < len(rs) && isDigit(rs[n]) {
				n++
			}
			return n
		}
		return 1
	}
	// punct run: " ?[^\s\p{L}\p{N}]+" with optional trailing [\r\n]* on
	// qwen2/llama3
	j = 0
	if r0 == ' ' && len(rs) > 1 && isPunct(rs[1]) {
		j = 1
	}
	if isPunct(rs[j]) {
		k = j
		for k < len(rs) && isPunct(rs[k]) {
			k++
		}
		if t.pre != preGPT2 {
			for k < len(rs) && (rs[k] == '\r' || rs[k] == '\n') {
				k++
			}
		}
		return k
	}
	// whitespace: \s*[\r\n]+ first (except gpt2), then \s+(?!\S) / \s+
	if isSpace(r0) {
		k := 0
		for k < len(rs) && isSpace(rs[k]) && rs[k] != '\r' && rs[k] != '\n' {
			k++
		}
		if t.pre != preGPT2 && k < len(rs) && (rs[k] == '\r' || rs[k] == '\n') {
			k++
			for k < len(rs) && (rs[k] == '\r' || rs[k] == '\n') {
				k++
			}
			return k
		}
		// \s+(?!\S) then \s+: if the run is followed by a non-space, leave
		// the final whitespace char to the next match
		k = 0
		for k < len(rs) && isSpace(rs[k]) {
			k++
		}
		if k < len(rs) && k > 1 {
			return k - 1
		}
		return k
	}
	return 1
}

// Encode turns text into token ids. Special tokens (control tokens such as
// <|im_start|>) are matched literally and are not reachable through merges.
func (t *Tokenizer) Encode(s string) []int {
	var ids []int
	mark := 0
	i := 0
	for i < len(s) {
		matched := false
		for _, sp := range t.specials {
			if strings.HasPrefix(s[i:], sp.s) {
				for _, piece := range t.split(s[mark:i]) {
					for _, id := range t.bpe(t.encodeBytes(piece)) {
						ids = append(ids, id)
					}
				}
				ids = append(ids, sp.id)
				i += len(sp.s)
				mark = i
				matched = true
				break
			}
		}
		if !matched {
			i++
		}
	}
	for _, piece := range t.split(s[mark:]) {
		for _, id := range t.bpe(t.encodeBytes(piece)) {
			ids = append(ids, id)
		}
	}
	return ids
}

// bpe merges an encoded piece down to vocab tokens.
func (t *Tokenizer) bpe(enc string) []int {
	syms := strings.Split(enc, "")
	for {
		best, rank := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.merges[[2]string{syms[i], syms[i+1]}]; ok && (rank < 0 || r < rank) {
				best, rank = i, r
			}
		}
		if best < 0 {
			break
		}
		syms[best] += syms[best+1]
		copy(syms[best+1:], syms[best+2:])
		syms = syms[:len(syms)-1]
	}
	ids := make([]int, 0, len(syms))
	for _, s := range syms {
		id, ok := t.vocab[s]
		if !ok {
			// fall back to single encoded characters
			for _, r := range s {
				if cid, ok := t.vocab[string(r)]; ok {
					ids = append(ids, cid)
				}
			}
			continue
		}
		ids = append(ids, id)
	}
	return ids
}

// Decode renders token ids back to text.
func (t *Tokenizer) Decode(ids []int) string {
	var b strings.Builder
	for _, id := range ids {
		b.WriteString(t.Text(id))
	}
	enc := b.String()
	var out strings.Builder
	out.Grow(len(enc))
	for _, r := range enc {
		if bb, ok := t.runeB[r]; ok {
			out.WriteByte(bb)
		} else {
			out.WriteRune(r)
		}
	}
	return out.String()
}

func (t *Tokenizer) String() string {
	return fmt.Sprintf("bpe vocab=%d merges=%d", len(t.tokens), len(t.merges))
}
