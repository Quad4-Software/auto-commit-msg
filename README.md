# auto-commit-msg

Generate a git commit message from the staged diff with a small local
language model. Pure Go, no cgo, no dependencies, single static binary.
Inference runs on CPU against a GGUF model.

## Build

    make                        # slim binary
    make fat MODEL=/path/m.gguf # single binary with the GGUF embedded

## Model

    auto-commit-msg setup       # download default model (Qwen3-0.6B Q4_K_M)

Or use any qwen2/qwen3 GGUF:

    auto-commit-msg -model /path/model.gguf
    export AUTO_COMMIT_MSG_MODEL=/path/model.gguf

Search order: -model flag, AUTO_COMMIT_MSG_MODEL, embedded blob,
$XDG_DATA_HOME/auto-commit-msg/model.gguf.

A GGUF whose general.basename contains "committed" gets the Committed
fine-tune prompt format and always emits a Conventional Commits line.
Other models get a generic prompt.

Architectures: qwen2, qwen3. Quants: F32, F16, Q4_0, Q5_0, Q8_0, Q4_K,
Q5_K, Q6_K.

## Usage

    git add -p
    auto-commit-msg               # print a suggested message
    auto-commit-msg -commit       # commit with it
    auto-commit-msg -edit         # commit, open the editor first
    auto-commit-msg -conventional # require type(scope): subject
    auto-commit-msg -all          # diff HEAD instead of staged only
    auto-commit-msg install-hook  # fill empty messages via prepare-commit-msg

## Performance

Ryzen 9 5900X, Qwen3-0.6B-class Q4_K_M, ctx 2048:

    diff      prompt tok   prefill     decode      peak RSS   total
    ~250 B    389          200 tok/s   21 tok/s    757 MB     ~2 s
    ~1.9 KB   944          176 tok/s   17 tok/s    990 MB     ~5.6 s
    ~67 KB    415          185 tok/s   19 tok/s    824 MB     ~2.7 s

Large diffs are compressed before prompting: lockfiles and generated
files collapse to headers, long file bodies truncate, omitted files are
listed by name.

Internals: mmap weights, growable KV cache, cache-tiled dequant +
batched matmul, AVX2 kernels with scalar fallback, fast exp, top-k
sampler. ACM_NO_ASM_DEQUANT=1 forces scalar dequant for debugging.

Env vars: AUTO_COMMIT_MSG_DIFF reads a diff from file instead of git.
ACM_TEST_MODEL sets the model for integration tests.

License: 0BSD.
