// Command rnabench roda o benchmark do AG-RNA (Trabalho 15) fora do navegador e
// salva o resultado em JSON. Útil pros tamanhos grandes, onde o modo "ingênuo"
// (sem as otimizações) leva minutos — aqui isso roda no terminal sem travar a UI.
//
// Uso:
//
//	go run ./cmd/rnabench -preset cheio
//	go run ./cmd/rnabench -preset amostra -out data/benchmarks/teste.json
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"sync"
	"time"

	"mlp-server/rnaga"
)

func main() {
	preset := flag.String("preset", "amostra", "amostra | media | cheio")
	out := flag.String("out", "", "arquivo JSON de saída (default: data/benchmarks/<preset>-<unix>.json)")
	flag.Parse()

	fmt.Printf("benchmark RNA+AG · preset=%q (executa do mais rápido pro mais lento; o ingênuo demora)\n", *preset)
	ch := make(chan rnaga.BenchModo, 8)
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		for m := range ch {
			fmt.Printf("  [%d] %-26s %9.1f ms   MSE=%.2f   hits=%d   workers=%d\n",
				m.Ordem, m.Nome, m.Ms, m.MelhorMSE, m.CacheHits, m.Workers)
		}
	}()
	res := rnaga.RodarBenchmarkPreset(ch, *preset)
	close(ch)
	wg.Wait()

	path := *out
	if path == "" {
		path = fmt.Sprintf("data/benchmarks/%s-%d.json", *preset, time.Now().Unix())
	}
	if err := os.MkdirAll("data/benchmarks", 0o755); err != nil {
		fmt.Println("erro ao criar diretório:", err)
		os.Exit(1)
	}
	data, _ := json.MarshalIndent(res, "", "  ")
	if err := os.WriteFile(path, data, 0o644); err != nil {
		fmt.Println("erro ao salvar:", err)
		os.Exit(1)
	}
	fmt.Printf("\nspeedup total (ingênuo → atual): %.1fx · maior diferença de MSE entre modos: %g\n",
		res.SpeedupTotal, res.MaxDiffMSE)
	fmt.Printf("salvo em %s\n", path)
}
