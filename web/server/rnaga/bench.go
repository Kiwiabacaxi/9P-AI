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
	Preset          string      `json:"preset"`
	Modos           []BenchModo `json:"modos"`
	NumCPU          int         `json:"numCpu"`
	SpeedupTotal    float64     `json:"speedupTotal"`
	MesmoMSE        bool        `json:"mesmoMse"`    // true = bit-idêntico nos modos
	MaxDiffMSE      float64     `json:"maxDiffMse"`  // maior diferença absoluta de MSE entre modos
	BenchCfg        Config      `json:"benchCfg"`
	FullCfg         Config      `json:"fullCfg"`
	FullIngenuoMs   float64     `json:"fullIngenuoMs"`   // estimativa: ingênuo no tamanho cheio
	FullOtimizadoMs float64     `json:"fullOtimizadoMs"` // estimativa: otimizado no tamanho cheio
	TimestampUnix   int64       `json:"timestampUnix"`
}

// BenchConfig — carga reduzida padrão (amostra).
func BenchConfig() Config { return BenchPreset("amostra") }

// BenchPreset — configs nomeadas de tamanho pro benchmark. "cheio" = 40×100×300
// (o tamanho real do enunciado; o modo ingênuo aqui leva ~15 min).
func BenchPreset(nome string) Config {
	switch nome {
	case "media", "média":
		return Config{PopSize: 24, MaxGeracoes: 25, ProbMutacao: 0.05, TetoEpocas: 200, Seed: 20260620}
	case "cheio":
		return Config{PopSize: 40, MaxGeracoes: 100, ProbMutacao: 0.05, TetoEpocas: 300, Seed: 20260620}
	default: // amostra
		return Config{PopSize: 16, MaxGeracoes: 8, ProbMutacao: 0.05, TetoEpocas: 150, Seed: 20260620}
	}
}

// trabalho — proxy do custo total (épocas treinadas) ∝ pop × gerações × épocas.
func trabalho(cfg Config) float64 {
	return float64(cfg.PopSize) * float64(cfg.MaxGeracoes) * float64(cfg.TetoEpocas)
}

// RodarBenchmark — executa os 5 modos cumulativos, mede cada um e (se
// progressCh != nil) emite cada modo conforme termina.
func RodarBenchmark(progressCh chan<- BenchModo, cfgBench Config) BenchResult {
	return rodarBenchmark(progressCh, cfgBench, "")
}

func rodarBenchmark(progressCh chan<- BenchModo, cfgBench Config, preset string) BenchResult {
	cfgBench = sanitizar(cfgBench)
	if cfgBench.Seed == 0 {
		cfgBench.Seed = 20260620
	}
	ncpu := runtime.NumCPU()

	defs := []struct {
		nome string
		opts execOpts
	}{
		{"ingênuo (laços + 1 core)", execOpts{workers: 1, usarMemo: false, onlineRealoca: true, offlineNaive: true}},
		{"+ matriz (gonum/BLAS)", execOpts{workers: 1, usarMemo: false, onlineRealoca: true, offlineNaive: false}},
		{"+ paralelo", execOpts{workers: ncpu, usarMemo: false, onlineRealoca: true, offlineNaive: false}},
		{"+ online sem alocação", execOpts{workers: ncpu, usarMemo: false, onlineRealoca: false, offlineNaive: false}},
		{"+ memoização (atual)", execOpts{workers: ncpu, usarMemo: true, onlineRealoca: false, offlineNaive: false}},
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

	full := BenchPreset("cheio")
	ratio := trabalho(full) / trabalho(cfgBench)
	ultimo := modos[len(modos)-1]
	speedup := 1.0
	if ultimo.Ms > 0 {
		speedup = modos[0].Ms / ultimo.Ms
	}
	// diferença de MSE relativa ao modo atual (otimizado): os modos com gonum
	// devem ser bit-idênticos; o ingênuo de laços pode divergir por FP ínfimo.
	maxDiff := 0.0
	for _, m := range modos {
		if d := math.Abs(m.MelhorMSE - ultimo.MelhorMSE); d > maxDiff {
			maxDiff = d
		}
	}

	return BenchResult{
		Preset:          preset,
		Modos:           modos,
		NumCPU:          ncpu,
		SpeedupTotal:    speedup,
		MesmoMSE:        maxDiff < 1e-6,
		MaxDiffMSE:      maxDiff,
		BenchCfg:        cfgBench,
		FullCfg:         full,
		FullIngenuoMs:   modos[0].Ms * ratio,
		FullOtimizadoMs: ultimo.Ms * ratio,
		TimestampUnix:   time.Now().Unix(),
	}
}

// RodarBenchmarkPreset — roda o benchmark num preset nomeado (amostra/media/cheio).
func RodarBenchmarkPreset(progressCh chan<- BenchModo, preset string) BenchResult {
	return rodarBenchmark(progressCh, BenchPreset(preset), preset)
}
