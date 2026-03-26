# CLI Reorganization Design

## Goal

Reorganize CLI programs into a unified `cli/` directory with clean naming, single `go.mod`, updated documentation, and validated builds.

## Current State

- 4 trabalhos in root-level directories with spaces/accents: `Trab 01/`, `Trab 02 - PT 1/`, `Trab 02 - PT 2/`, `Trab 03/`
- 2 MLP challenges in `Desafios MLP/` subdirectories
- Each has its own `go.mod` with overlapping dependencies (Charm stack)
- Artifact files to remove: `Trab 01/appemrust.rs`, `Trab 01/cli/old_style/`

## Target Structure

```
cli/
├── go.mod                    # Unified module
├── go.sum
├── README.md                 # Overview with links to each program
├── trab01-hebb/
│   ├── main.go
│   └── README.md
├── trab02-perceptron-letras/
│   ├── main.go
│   ├── tui.go
│   └── README.md
├── trab02-perceptron-portas/
│   ├── main.go
│   ├── tui.go
│   └── README.md
├── trab03-madaline/
│   ├── main.go
│   ├── tui.go
│   └── README.md
├── desafio-mlp/
│   ├── main.go
│   ├── tui.go
│   └── README.md
└── desafio-mlp-letras/
    ├── main.go
    ├── tui.go
    └── README.md
```

## Changes

### 1. Directory moves

| From | To |
|------|----|
| `Trab 01/cli/main.go` | `cli/trab01-hebb/main.go` |
| `Trab 02 - PT 1/cli/{main.go,tui.go}` | `cli/trab02-perceptron-letras/{main.go,tui.go}` |
| `Trab 02 - PT 2/cli/{main.go,tui.go}` | `cli/trab02-perceptron-portas/{main.go,tui.go}` |
| `Trab 03/cli/{main.go,tui.go}` | `cli/trab03-madaline/{main.go,tui.go}` |
| `Desafios MLP/Desafio Multilayer Perceptron (MLP)/cli/{main.go,tui.go}` | `cli/desafio-mlp/{main.go,tui.go}` |
| `Desafios MLP/Multilayer Perceptron (MLP) Letras/cli/{main.go,tui.go}` | `cli/desafio-mlp-letras/{main.go,tui.go}` |

### 2. Deletions

- `Trab 01/appemrust.rs`
- `Trab 01/cli/old_style/` (entire directory)
- All old `go.mod` / `go.sum` files (replaced by unified one)
- All old empty parent directories after move

### 3. Unified go.mod

- Module path: `github.com/kiwiabacaxi/9P-AI/cli` (or matching existing convention)
- Dependencies: Charm stack (bubbletea, lipgloss, bubbles)
- Each subdir remains `package main` — run via `go run ./trab01-hebb/`

### 4. Documentation updates

- **Root `README.md`**: Overview of project with links to `cli/README.md` and `web/` section
- **`cli/README.md`**: Overview table with links to each program's README
- **Individual READMEs**: Moved from old trab READMEs, updated paths and run commands

### 5. Validation

- `go build ./...` from `cli/` — all 6 programs compile
- `GOOS=js GOARCH=wasm go build` from `web/server/` — WASM still builds
- Web server `go build` from `web/server/` — still builds

## Not Changed

- `web/` directory (untouched)
- `slides/` directory
- `docs/` directory
- GitHub Actions workflow
- `.gitignore` (may need minor path updates)
