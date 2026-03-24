PORT := 8080

.PHONY: run build clean kill

run: build kill
	@echo "» iniciando servidor em http://localhost:$(PORT)"
	@cd server && ./mlp-server & \
	for i in $$(seq 1 20); do \
		curl -s http://localhost:$(PORT)/api/status > /dev/null && break; \
		sleep 0.2; \
	done; \
	open http://localhost:$(PORT); \
	wait

build:
	@echo "» compilando..."
	@cd server && go build -o mlp-server .

kill:
	@lsof -ti tcp:$(PORT) | xargs kill -9 2>/dev/null || true

clean: kill
	@rm -f server/mlp-server
