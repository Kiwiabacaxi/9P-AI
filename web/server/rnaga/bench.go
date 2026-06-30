package rnaga

import (
	"math"
	"runtime"
	"time"
)

// =============================================================================
// Benchmark — mede, de verdade, o efeito de cada técnica de otimização.
//
// Roda o MESMO AG (mesma seed, mesmas arquiteturas) sob 4 modos cumulativos. Como
// nenhuma das técnicas muda O QUE é computado (só QUÃO RÁPIDO), o MSE final sai
// idêntico nos 4 modos — é a prova de que medimos o mesmo trabalho, só mais rápido.
//
//   0) ingênuo (sequencial)      — 1 core, sem cache, online realocando buffers
//   1) + paralelo                — N cores (goroutines)
//   2) + online sem alocação     — buffers reaproveitados (0 alloc no loop)
//   3) + memoização (atual)      — cache de fitness por arquitetura
//
// Roda numa carga REDUZIDA (senão o modo ingênuo levaria minutos) e extrapola
// linearmente pro tamanho cheio (40×100), conectando com o "vai demorar muito".
// =============================================================================

type BenchModo struct {
	Ordem     int     `json:"ordem"` // posição canônica (0=ingênuo … 3=atual)
	Nome      string  `json:"nome"`
	Ms        float64 `json:"ms"`
	MelhorMSE float64 `json:"melhorMse"`
	CacheHits int     `json:"cacheHits"`
	Workers   int     `json:"workers"`
}

type BenchResult struct {
	Modos           []BenchModo `json:"modos"`
	NumCPU          int         `json:"numCpu"`
	SpeedupTotal    float64     `json:"speedupTotal"`
	MesmoMSE        bool        `json:"mesmoMse"`
	BenchCfg        Config      `json:"benchCfg"`
	FullCfg         Config      `json:"fullCfg"`
	FullIngenuoMs   float64     `json:"fullIngenuoMs"`   // estimativa: ingênuo no tamanho cheio
	FullOtimizadoMs float64     `json:"fullOtimizadoMs"` // estimativa: otimizado no tamanho cheio
}

// BenchConfig — carga reduzida do benchmark (pra o modo ingênuo não levar minutos).
func BenchConfig() Config {
	return Config{PopSize: 16, MaxGeracoes: 8, ProbMutacao: 0.05, TetoEpocas: 150, Seed: 20260620}
}

// trabalho — proxy do custo total (épocas treinadas) ∝ pop × gerações × épocas.
func trabalho(cfg Config) float64 {
	return float64(cfg.PopSize) * float64(cfg.MaxGeracoes) * float64(cfg.TetoEpocas)
}

// RodarBenchmark — executa os 4 modos, mede cada um e (se progressCh != nil)
// emite cada modo conforme termina.
func RodarBenchmark(progressCh chan<- BenchModo, cfgBench Config) BenchResult {
	cfgBench = sanitizar(cfgBench)
	if cfgBench.Seed == 0 {
		cfgBench.Seed = 20260620
	}
	ncpu := runtime.NumCPU()

	defs := []struct {
		nome string
		opts execOpts
	}{
		{"ingênuo (sequencial)", execOpts{workers: 1, usarMemo: false, onlineRealoca: true}},
		{"+ paralelo", execOpts{workers: ncpu, usarMemo: false, onlineRealoca: true}},
		{"+ online sem alocação", execOpts{workers: ncpu, usarMemo: false, onlineRealoca: false}},
		{"+ memoização (atual)", execOpts{workers: ncpu, usarMemo: true, onlineRealoca: false}},
	}

	// Executa do mais RÁPIDO (último) pro mais LENTO (ingênuo), pra as barras
	// aparecerem logo na UI; cada modo carrega sua Ordem canônica pra exibição.
	modos := make([]BenchModo, len(defs))
	for k := len(defs) - 1; k >= 0; k-- {
		d := defs[k]
		t0 := time.Now()
		res, hits := treinarComOpts(nil, cfgBench, d.opts)
		ms := float64(time.Since(t0).Microseconds()) / 1000.0
		m := BenchModo{Ordem: k, Nome: d.nome, Ms: ms, MelhorMSE: res.MelhorMSE, CacheHits: hits, Workers: d.opts.workers}
		modos[k] = m
		if progressCh != nil {
			progressCh <- m
		}
	}

	full := DefaultConfig()
	ratio := trabalho(full) / trabalho(cfgBench)
	ultimo := modos[len(modos)-1]
	speedup := 1.0
	if ultimo.Ms > 0 {
		speedup = modos[0].Ms / ultimo.Ms
	}
	mesmo := true
	for _, m := range modos {
		if math.Abs(m.MelhorMSE-modos[0].MelhorMSE) > 1e-6 {
			mesmo = false
		}
	}

	return BenchResult{
		Modos:           modos,
		NumCPU:          ncpu,
		SpeedupTotal:    speedup,
		MesmoMSE:        mesmo,
		BenchCfg:        cfgBench,
		FullCfg:         full,
		FullIngenuoMs:   modos[0].Ms * ratio,
		FullOtimizadoMs: ultimo.Ms * ratio,
	}
}
