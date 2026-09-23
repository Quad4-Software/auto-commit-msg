BIN = auto-commit-msg
# MODEL is the GGUF bundled into the fat build. Point it at any qwen2/qwen3
# GGUF: make fat MODEL=/path/to/model.gguf
MODEL = model.gguf

all: $(BIN)

$(BIN):
	CGO_ENABLED=0 go build -trimpath -o $(BIN) .

fat:
	[ "$(MODEL)" = "model.gguf" ] || cp -f "$(MODEL)" model.gguf
	[ -f model.gguf ] || { echo "no model: set MODEL=/path/to/model.gguf" >&2; exit 1; }
	CGO_ENABLED=0 go build -trimpath -tags bundle -o $(BIN)-fat .

test:
	go test ./...

vet:
	go vet ./...

clean:
	rm -f $(BIN) $(BIN)-fat

distclean: clean
	rm -f model.gguf

.PHONY: all fat test vet clean distclean
