PORT := 8080

.PHONY: run build build-frontend build-server clean kill

run: build kill
	@echo "» iniciando servidor em http://localhost:$(PORT)"
	@cd server && ./mlp-server & \
	for i in $$(seq 1 20); do \
		curl -s http://localhost:$(PORT)/api/status > /dev/null && break; \
		sleep 0.2; \
	done; \
	open http://localhost:$(PORT); \
	wait

build: build-frontend build-server

build-frontend:
	@echo "» compilando frontend..."
	@cd frontend && npm run build

build-server:
	@echo "» compilando servidor..."
	@cd server && go build -o mlp-server .

build-windows:
	@echo "» compilando para Windows..."
	@cd server && GOOS=windows GOARCH=amd64 go build -o mlp-server.exe .

kill:
	@lsof -ti tcp:$(PORT) | xargs kill -9 2>/dev/null || true

clean: kill
	@rm -f server/mlp-server server/mlp-server.exe
