# =============================================================================
# 9P-AI — Makefile cross-platform (Windows + Mac/Linux)
# =============================================================================

.PHONY: build run dev clean frontend server help

# Detectar OS
ifeq ($(OS),Windows_NT)
    PYTHON := $(firstword $(wildcard C:/Users/*/AppData/Local/Programs/Python/Python3*/python.exe) python)
    RM := del /q /f
    RMDIR := rmdir /s /q
else
    PYTHON := python3
    RM := rm -f
    RMDIR := rm -rf
endif

help: ## Mostra este help
	@echo "9P-AI — Comandos disponiveis:"
	@echo ""
	@echo "  make build     Build frontend + backend"
	@echo "  make run       Build tudo e inicia o servidor (localhost:8080)"
	@echo "  make dev       Inicia Vite dev server (localhost:5173) + Go backend"
	@echo "  make frontend  Build apenas o frontend"
	@echo "  make server    Build apenas o backend Go"
	@echo "  make clean     Limpa artefatos de build"
	@echo ""

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

frontend: ## Build frontend (npm run build)
	cd web/frontend && npm run build

server: ## Build backend Go
	cd web/server && go build -o server$(if $(filter Windows_NT,$(OS)),.exe,) .

build: frontend server ## Build frontend + backend

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

run: frontend ## Build frontend e inicia servidor Go
	cd web/server && go run .

dev: ## Dev mode: Vite (5173) + Go (8080) em paralelo
	@echo "Iniciando Go backend..."
	cd web/server && go run . &
	@echo "Iniciando Vite dev server..."
	cd web/frontend && npm run dev

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean: ## Limpa artefatos
	$(RM) web/server/server web/server/server.exe
	$(RMDIR) web/static/assets 2>/dev/null || true
	$(RM) web/static/index.html 2>/dev/null || true
