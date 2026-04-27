# =============================================================================
# 9P-AI — Makefile cross-platform (Windows + Mac/Linux)
# =============================================================================
# Uso tipico:
#   make run       — builda tudo e sobe o servidor em http://localhost:8080
#   make install   — instala deps do frontend (npm install)
# =============================================================================

PORT := 8080

.PHONY: help build run dev clean frontend server install

# Detectar OS
ifeq ($(OS),Windows_NT)
    SERVER_BIN := mlp-server.exe
    RM := del /q /f
    RMDIR := rmdir /s /q
else
    SERVER_BIN := mlp-server
    RM := rm -f
    RMDIR := rm -rf
endif

ifeq ($(OS),Windows_NT)
help:
	@echo 9P-AI - Comandos disponiveis:
	@echo.
	@echo   make run       Build tudo e inicia o servidor (http://localhost:$(PORT))
	@echo   make build     Build frontend + backend
	@echo   make install   Instala deps do frontend (npm install)
	@echo   make dev       Dev mode: Vite (5173) + Go backend (8080)
	@echo   make frontend  Build apenas o frontend
	@echo   make server    Build apenas o backend Go
	@echo   make clean     Limpa artefatos de build
	@echo.
else
help:
	@echo "9P-AI — Comandos disponiveis:"
	@echo ""
	@echo "  make run       Build tudo e inicia o servidor (http://localhost:$(PORT))"
	@echo "  make build     Build frontend + backend"
	@echo "  make install   Instala deps do frontend (npm install)"
	@echo "  make dev       Dev mode: Vite (5173) + Go backend (8080)"
	@echo "  make frontend  Build apenas o frontend"
	@echo "  make server    Build apenas o backend Go"
	@echo "  make clean     Limpa artefatos de build"
	@echo ""
endif

# ---------------------------------------------------------------------------
# Dependencies — npm install roda so quando package.json e mais novo que node_modules
# ---------------------------------------------------------------------------

web/frontend/node_modules: web/frontend/package.json
	cd web/frontend && npm install

install: web/frontend/node_modules

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

frontend: web/frontend/node_modules
	cd web/frontend && npm run build

server:
	cd web/server && go build -o $(SERVER_BIN) .

build: frontend server

# ---------------------------------------------------------------------------
# Run — delega pro script de cada OS (kill porta + start + wait + open browser)
# ---------------------------------------------------------------------------

ifeq ($(OS),Windows_NT)

run: build
	cd web && powershell -NoProfile -ExecutionPolicy Bypass -File run.ps1

else

run: build
	@$(MAKE) -s -C web run

endif

# ---------------------------------------------------------------------------
# Dev mode (Vite HMR)
# ---------------------------------------------------------------------------

dev: web/frontend/node_modules
	@echo "Iniciando Go backend..."
	cd web/server && go run . &
	@echo "Iniciando Vite dev server..."
	cd web/frontend && npm run dev

# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------

clean:
	-$(RM) web/server/mlp-server 2>/dev/null
	-$(RM) web/server/mlp-server.exe 2>/dev/null
	-$(RM) web/server/server 2>/dev/null
	-$(RM) web/server/server.exe 2>/dev/null
	-$(RMDIR) web/static/assets 2>/dev/null
	-$(RM) web/static/index.html 2>/dev/null
