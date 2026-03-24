package imgreg_bench

import (
	"context"
	"sync"
	"time"

	std "mlp-server/imgreg"
	gor "mlp-server/imgreg_goroutines"
	mat "mlp-server/imgreg_matrix"
	mb  "mlp-server/imgreg_minibatch"
)

type BenchConfig struct {
	HiddenLayers    int     `json:"hiddenLayers"`
	NeuronsPerLayer int     `json:"neuronsPerLayer"`
	LearningRate    float64 `json:"learningRate"`
	Imagem          string  `json:"imagem"`
	MaxEpocas       int     `json:"maxEpocas"`
	BatchSize       int     `json:"batchSize"`
	NumWorkers      int     `json:"numWorkers"`
	Parallel        bool    `json:"parallel"`
}

// Step é uma cópia local do Step dos backends (sem import circular)
type Step struct {
	Epoca         int          `json:"epoca"`
	MaxEpocas     int          `json:"maxEpocas"`
	Loss          float64      `json:"loss"`
	OutputPixels  [][3]float64 `json:"outputPixels"`
	ActiveLayer   int          `json:"activeLayer"`
	Done          bool         `json:"done"`
	Convergiu     bool         `json:"convergiu"`
	LossHistorico []float64    `json:"lossHistorico"`
	ElapsedMs     int64        `json:"elapsedMs"`
	EpochMs       int64        `json:"epochMs"`
}

type BenchStep struct {
	Backend string `json:"backend"` // "standard" | "goroutines" | "matrix" | "minibatch"
	Step    Step   `json:"step"`
}

func fromStd(s std.Step, elapsedMs, epochMs int64) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: elapsedMs, EpochMs: epochMs}
}
func fromGor(s gor.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}
func fromMat(s mat.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}
func fromMb(s mb.Step) Step {
	return Step{Epoca: s.Epoca, MaxEpocas: s.MaxEpocas, Loss: s.Loss,
		OutputPixels: s.OutputPixels, ActiveLayer: s.ActiveLayer, Done: s.Done,
		Convergiu: s.Convergiu, LossHistorico: s.LossHistorico,
		ElapsedMs: s.ElapsedMs, EpochMs: s.EpochMs}
}

func Rodar(ctx context.Context, cfg BenchConfig, ch chan<- BenchStep) {
	stdCfg := std.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas}
	gorCfg := gor.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas}
	matCfg := mat.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas}
	mbCfg := mb.Config{HiddenLayers: cfg.HiddenLayers, NeuronsPerLayer: cfg.NeuronsPerLayer,
		LearningRate: cfg.LearningRate, Imagem: cfg.Imagem, MaxEpocas: cfg.MaxEpocas,
		BatchSize: cfg.BatchSize, NumWorkers: cfg.NumWorkers}

	// Standard wraps the original imgreg which has no timing fields — measure externally
	runStd := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan std.Step, 64)
		start := time.Now()
		go std.Treinar(ctx, stdCfg, c)
		var lastEpochStart time.Time
		var epochMs int64
		for s := range c {
			now := time.Now()
			if !s.Done {
				epochMs = now.Sub(lastEpochStart).Milliseconds()
				lastEpochStart = now
			}
			select {
			case out <- BenchStep{Backend: "standard", Step: fromStd(s, time.Since(start).Milliseconds(), epochMs)}:
			case <-ctx.Done():
				return
			}
		}
	}
	runGor := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan gor.Step, 64)
		go gor.Treinar(ctx, gorCfg, c)
		for s := range c {
			select {
			case out <- BenchStep{Backend: "goroutines", Step: fromGor(s)}:
			case <-ctx.Done():
				return
			}
		}
	}
	runMat := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan mat.Step, 64)
		go mat.Treinar(ctx, matCfg, c)
		for s := range c {
			select {
			case out <- BenchStep{Backend: "matrix", Step: fromMat(s)}:
			case <-ctx.Done():
				return
			}
		}
	}
	runMb := func(ctx context.Context, out chan<- BenchStep) {
		c := make(chan mb.Step, 64)
		go mb.Treinar(ctx, mbCfg, c)
		for s := range c {
			select {
			case out <- BenchStep{Backend: "minibatch", Step: fromMb(s)}:
			case <-ctx.Done():
				return
			}
		}
	}

	runners := []func(context.Context, chan<- BenchStep){runStd, runGor, runMat, runMb}

	if cfg.Parallel {
		proxy := make(chan BenchStep, 256)
		var wg sync.WaitGroup
		for _, fn := range runners {
			wg.Add(1)
			fn := fn
			go func() {
				defer wg.Done()
				fn(ctx, proxy)
			}()
		}
		go func() {
			wg.Wait()
			close(proxy)
		}()
		for bs := range proxy {
			select {
			case ch <- bs:
			case <-ctx.Done():
				return
			}
		}
	} else {
		// Sequencial: standard → goroutines → matrix → minibatch
		for _, fn := range runners {
			fn(ctx, ch)
		}
	}
	close(ch)
}
