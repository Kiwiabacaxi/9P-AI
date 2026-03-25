// +build js,wasm

package main

import (
	"context"
	"encoding/json"
	"sync"
	"syscall/js"

	"mlp-server/hebb"
	"mlp-server/imgreg"
	igoroutines "mlp-server/imgreg_goroutines"
	iminibatch "mlp-server/imgreg_minibatch"
	"mlp-server/letras"
	"mlp-server/madaline"
	"mlp-server/mlp"
	perceptronletras "mlp-server/perceptron_letras"
	perceptronportas "mlp-server/perceptron_portas"
)

// ═══════════════════════════════════════════════════════════════════════
// Estado global (mirrors main.go)
// ═══════════════════════════════════════════════════════════════════════

var (
	mu sync.RWMutex

	mlpRede *mlp.MLP
	mlpRes  *mlp.MLPResult

	ltrRede     *letras.LtrMLP
	ltrRes      *letras.LtrResult
	ltrTraining bool

	hebbRedes       map[string]*hebb.HebbResult
	percPortasRedes map[string]*perceptronportas.PercPortasResult
	percLetrasRede  *perceptronletras.PercLetrasResult

	madRede     *madaline.MadNet
	madRes      *madaline.MadResult
	madTraining bool

	imgregRede     *imgreg.Net
	imgregTraining bool
	imgregCancel   context.CancelFunc

	igorRede     *igoroutines.Net
	igorTraining bool
	igorCancel   context.CancelFunc

	imbRede     *iminibatch.Net
	imbTraining bool
	imbCancel   context.CancelFunc
)

func toJSON(v any) string {
	b, _ := json.Marshal(v)
	return string(b)
}

// ═══════════════════════════════════════════════════════════════════════
// Status
// ═══════════════════════════════════════════════════════════════════════

func wasmStatus(_ js.Value, _ []js.Value) any {
	mu.RLock()
	defer mu.RUnlock()
	return toJSON(map[string]any{
		"mlpTrained":     mlpRede != nil,
		"letrasTrained":  ltrRede != nil,
		"ltrTraining":    ltrTraining,
		"hebbCount":      len(hebbRedes),
		"percPortaCount": len(percPortasRedes),
		"percLetrasDone": percLetrasRede != nil,
		"madTrained":     madRede != nil,
		"madTraining":    madTraining,
		"imgregTrained":  imgregRede != nil,
		"imgregTraining": imgregTraining,
	})
}

// ═══════════════════════════════════════════════════════════════════════
// Hebb
// ═══════════════════════════════════════════════════════════════════════

func wasmHebbPortas(_ js.Value, _ []js.Value) any {
	mu.RLock()
	defer mu.RUnlock()
	type portaInfo struct {
		Nome    string `json:"nome"`
		Desc    string `json:"desc"`
		Trained bool   `json:"trained"`
	}
	var result []portaInfo
	for _, p := range hebb.PortasLogicas() {
		_, trained := hebbRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Desc: p.Desc, Trained: trained})
	}
	return toJSON(result)
}

func wasmHebbTrain(_ js.Value, args []js.Value) any {
	porta := args[0].String()
	portas := hebb.PortasLogicas()

	if porta == "ALL" || porta == "" {
		results := make(map[string]*hebb.HebbResult)
		for _, p := range portas {
			res := hebb.Treinar(p)
			results[p.Nome] = &res
		}
		mu.Lock()
		for k, v := range results {
			hebbRedes[k] = v
		}
		mu.Unlock()
		return toJSON(results)
	}

	for _, p := range portas {
		if p.Nome == porta {
			res := hebb.Treinar(p)
			mu.Lock()
			hebbRedes[p.Nome] = &res
			mu.Unlock()
			return toJSON(res)
		}
	}
	return toJSON(map[string]string{"error": "porta inválida: " + porta})
}

// ═══════════════════════════════════════════════════════════════════════
// Perceptron Portas
// ═══════════════════════════════════════════════════════════════════════

func wasmPercPortasLista(_ js.Value, _ []js.Value) any {
	mu.RLock()
	defer mu.RUnlock()
	type portaInfo struct {
		Nome    string `json:"nome"`
		Trained bool   `json:"trained"`
	}
	var result []portaInfo
	for _, p := range perceptronportas.PortasLogicas() {
		_, trained := percPortasRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Trained: trained})
	}
	return toJSON(result)
}

func wasmPercPortasTrain(_ js.Value, args []js.Value) any {
	porta := args[0].String()
	portas := perceptronportas.PortasLogicas()

	if porta == "ALL" || porta == "" {
		results := make(map[string]*perceptronportas.PercPortasResult)
		for _, p := range portas {
			res := perceptronportas.Treinar(p)
			results[p.Nome] = &res
		}
		mu.Lock()
		for k, v := range results {
			percPortasRedes[k] = v
		}
		mu.Unlock()
		return toJSON(results)
	}

	for _, p := range portas {
		if p.Nome == porta {
			res := perceptronportas.Treinar(p)
			mu.Lock()
			percPortasRedes[p.Nome] = &res
			mu.Unlock()
			return toJSON(res)
		}
	}
	return toJSON(map[string]string{"error": "porta inválida: " + porta})
}

// ═══════════════════════════════════════════════════════════════════════
// Perceptron Letras
// ═══════════════════════════════════════════════════════════════════════

func wasmPercLetrasTrain(_ js.Value, _ []js.Value) any {
	res := perceptronletras.Treinar()
	mu.Lock()
	percLetrasRede = &res
	mu.Unlock()
	return toJSON(res)
}

func wasmPercLetrasDataset(_ js.Value, _ []js.Value) any {
	return toJSON([]perceptronletras.DatasetResp{
		{Letra: "A", Grade: perceptronletras.LetraA()},
		{Letra: "B", Grade: perceptronletras.LetraB()},
	})
}

// ═══════════════════════════════════════════════════════════════════════
// MLP Desafio
// ═══════════════════════════════════════════════════════════════════════

func wasmMlpTrain(_ js.Value, _ []js.Value) any {
	return promiseGo(func() string {
		res := mlp.Treinar()
		mu.Lock()
		mlpRes = &res
		rede := res.Rede
		mlpRede = &rede
		mu.Unlock()
		return toJSON(res)
	})
}

func wasmMlpResult(_ js.Value, _ []js.Value) any {
	mu.RLock()
	res := mlpRes
	mu.RUnlock()
	if res == nil {
		return toJSON(map[string]string{"error": "rede não treinada"})
	}
	return toJSON(res)
}

// ═══════════════════════════════════════════════════════════════════════
// MLP Letras
// ═══════════════════════════════════════════════════════════════════════

func wasmLetrasTrain(_ js.Value, args []js.Value) any {
	onStep := args[0]
	mu.Lock()
	if ltrTraining {
		mu.Unlock()
		return nil
	}
	ltrTraining = true
	mu.Unlock()

	go func() {
		ch := make(chan letras.LtrStep, 64)
		go func() {
			res, rede := letras.Treinar(ch)
			mu.Lock()
			ltrRes = &res
			ltrRede = &rede
			ltrTraining = false
			mu.Unlock()
			close(ch)
		}()
		for step := range ch {
			onStep.Invoke(toJSON(step))
		}
		if ltrRes != nil {
			onStep.Invoke(toJSON(map[string]any{"done": true, "result": ltrRes}))
		}
	}()
	return nil
}

func wasmLetrasResult(_ js.Value, _ []js.Value) any {
	mu.RLock()
	res := ltrRes
	mu.RUnlock()
	if res == nil {
		return toJSON(map[string]string{"error": "rede não treinada"})
	}
	return toJSON(res)
}

func wasmLetrasClassify(_ js.Value, args []js.Value) any {
	mu.RLock()
	rede := ltrRede
	mu.RUnlock()
	if rede == nil {
		return toJSON(map[string]string{"error": "rede não treinada"})
	}
	var req letras.ClassifyReq
	json.Unmarshal([]byte(args[0].String()), &req)
	return toJSON(letras.Classificar(*rede, req.Grade))
}

func wasmLetrasDataset(_ js.Value, _ []js.Value) any {
	dataset := letras.Dataset()
	type entry struct {
		Letra string              `json:"letra"`
		Idx   int                 `json:"idx"`
		Grade [letras.NIn]float64 `json:"grade"`
	}
	entries := make([]entry, letras.NOut)
	for i := 0; i < letras.NOut; i++ {
		entries[i] = entry{Letra: letras.Nomes[i], Idx: i, Grade: dataset[i]}
	}
	return toJSON(entries)
}

// ═══════════════════════════════════════════════════════════════════════
// MADALINE
// ═══════════════════════════════════════════════════════════════════════

func wasmMadTrain(_ js.Value, args []js.Value) any {
	onStep := args[0]
	mu.Lock()
	if madTraining {
		mu.Unlock()
		return nil
	}
	madTraining = true
	mu.Unlock()

	go func() {
		ch := make(chan madaline.MadStep, 64)
		go func() {
			res, rede := madaline.Treinar(ch)
			mu.Lock()
			madRes = &res
			madRede = &rede
			madTraining = false
			mu.Unlock()
			close(ch)
		}()
		for step := range ch {
			onStep.Invoke(toJSON(step))
		}
		if madRes != nil {
			onStep.Invoke(toJSON(map[string]any{"done": true, "result": madRes}))
		}
	}()
	return nil
}

func wasmMadResult(_ js.Value, _ []js.Value) any {
	mu.RLock()
	res := madRes
	mu.RUnlock()
	if res == nil {
		return toJSON(map[string]string{"error": "rede não treinada"})
	}
	return toJSON(res)
}

func wasmMadClassify(_ js.Value, args []js.Value) any {
	mu.RLock()
	rede := madRede
	mu.RUnlock()
	if rede == nil {
		return toJSON(map[string]string{"error": "rede não treinada"})
	}
	var req madaline.MadClassifyReq
	json.Unmarshal([]byte(args[0].String()), &req)
	return toJSON(madaline.Classificar(*rede, req.Grade))
}

func wasmMadDataset(_ js.Value, _ []js.Value) any {
	dataset := madaline.Dataset()
	type entry struct {
		Letra string                `json:"letra"`
		Idx   int                   `json:"idx"`
		Grade [madaline.NIn]float64 `json:"grade"`
	}
	entries := make([]entry, madaline.NLetras)
	for i := 0; i < madaline.NLetras; i++ {
		var g [madaline.NIn]float64
		for j := 0; j < madaline.NIn; j++ {
			g[j] = float64(dataset[i][j])
		}
		entries[i] = entry{Letra: madaline.Nomes[i], Idx: i, Grade: g}
	}
	return toJSON(entries)
}

// ═══════════════════════════════════════════════════════════════════════
// Image Regression — Standard
// ═══════════════════════════════════════════════════════════════════════

func wasmImgregTarget(_ js.Value, args []js.Value) any {
	img := "coracao"
	if len(args) > 0 && args[0].String() != "" {
		img = args[0].String()
	}
	return toJSON(imgreg.GetTarget(img))
}

func wasmImgregTrain(_ js.Value, args []js.Value) any {
	cfgJSON := args[0].String()
	onStep := args[1]

	mu.Lock()
	if imgregTraining {
		mu.Unlock()
		return nil
	}
	imgregTraining = true
	mu.Unlock()

	go func() {
		var cfg imgreg.Config
		json.Unmarshal([]byte(cfgJSON), &cfg)

		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		imgregCancel = cancel
		mu.Unlock()

		ch := make(chan imgreg.Step, 64)
		go func() {
			rede := imgreg.Treinar(ctx, cfg, ch)
			mu.Lock()
			imgregRede = &rede
			imgregTraining = false
			imgregCancel = nil
			mu.Unlock()
		}()
		for step := range ch {
			onStep.Invoke(toJSON(step))
		}
	}()
	return nil
}

func wasmImgregReset(_ js.Value, _ []js.Value) any {
	mu.Lock()
	if imgregCancel != nil {
		imgregCancel()
		imgregCancel = nil
	}
	imgregRede = nil
	imgregTraining = false
	mu.Unlock()
	return toJSON(map[string]string{"ok": "resetado"})
}

// ═══════════════════════════════════════════════════════════════════════
// Image Regression — Goroutines
// ═══════════════════════════════════════════════════════════════════════

func wasmIgorTrain(_ js.Value, args []js.Value) any {
	cfgJSON := args[0].String()
	onStep := args[1]

	mu.Lock()
	if igorTraining {
		mu.Unlock()
		return nil
	}
	igorTraining = true
	mu.Unlock()

	go func() {
		var cfg igoroutines.Config
		json.Unmarshal([]byte(cfgJSON), &cfg)

		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		igorCancel = cancel
		mu.Unlock()

		ch := make(chan igoroutines.Step, 64)
		go func() {
			rede := igoroutines.Treinar(ctx, cfg, ch)
			mu.Lock()
			igorRede = &rede
			igorTraining = false
			igorCancel = nil
			mu.Unlock()
		}()
		for step := range ch {
			onStep.Invoke(toJSON(step))
		}
	}()
	return nil
}

func wasmIgorReset(_ js.Value, _ []js.Value) any {
	mu.Lock()
	if igorCancel != nil {
		igorCancel()
		igorCancel = nil
	}
	igorRede = nil
	igorTraining = false
	mu.Unlock()
	return toJSON(map[string]string{"ok": "resetado"})
}

// ═══════════════════════════════════════════════════════════════════════
// Image Regression — Minibatch
// ═══════════════════════════════════════════════════════════════════════

func wasmImbTrain(_ js.Value, args []js.Value) any {
	cfgJSON := args[0].String()
	onStep := args[1]

	mu.Lock()
	if imbTraining {
		mu.Unlock()
		return nil
	}
	imbTraining = true
	mu.Unlock()

	go func() {
		var cfg iminibatch.Config
		json.Unmarshal([]byte(cfgJSON), &cfg)

		ctx, cancel := context.WithCancel(context.Background())
		mu.Lock()
		imbCancel = cancel
		mu.Unlock()

		ch := make(chan iminibatch.Step, 64)
		go func() {
			rede := iminibatch.Treinar(ctx, cfg, ch)
			mu.Lock()
			imbRede = &rede
			imbTraining = false
			imbCancel = nil
			mu.Unlock()
		}()
		for step := range ch {
			onStep.Invoke(toJSON(step))
		}
	}()
	return nil
}

func wasmImbReset(_ js.Value, _ []js.Value) any {
	mu.Lock()
	if imbCancel != nil {
		imbCancel()
		imbCancel = nil
	}
	imbRede = nil
	imbTraining = false
	mu.Unlock()
	return toJSON(map[string]string{"ok": "resetado"})
}

// ═══════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════

// promiseGo wraps a heavy Go function in a JS Promise so it doesn't block the event loop.
func promiseGo(fn func() string) any {
	handler := js.FuncOf(func(_ js.Value, resolveReject []js.Value) any {
		resolve := resolveReject[0]
		go func() {
			result := fn()
			resolve.Invoke(result)
		}()
		return nil
	})
	return js.Global().Get("Promise").New(handler)
}

// ═══════════════════════════════════════════════════════════════════════
// main — registers all functions and blocks forever
// ═══════════════════════════════════════════════════════════════════════

func main() {
	hebbRedes = make(map[string]*hebb.HebbResult)
	percPortasRedes = make(map[string]*perceptronportas.PercPortasResult)

	g := js.Global()

	// Status
	g.Set("wasmStatus", js.FuncOf(wasmStatus))

	// Hebb
	g.Set("wasmHebbPortas", js.FuncOf(wasmHebbPortas))
	g.Set("wasmHebbTrain", js.FuncOf(wasmHebbTrain))

	// Perceptron Portas
	g.Set("wasmPercPortasLista", js.FuncOf(wasmPercPortasLista))
	g.Set("wasmPercPortasTrain", js.FuncOf(wasmPercPortasTrain))

	// Perceptron Letras
	g.Set("wasmPercLetrasTrain", js.FuncOf(wasmPercLetrasTrain))
	g.Set("wasmPercLetrasDataset", js.FuncOf(wasmPercLetrasDataset))

	// MLP Desafio
	g.Set("wasmMlpTrain", js.FuncOf(wasmMlpTrain))
	g.Set("wasmMlpResult", js.FuncOf(wasmMlpResult))

	// MLP Letras
	g.Set("wasmLetrasTrain", js.FuncOf(wasmLetrasTrain))
	g.Set("wasmLetrasResult", js.FuncOf(wasmLetrasResult))
	g.Set("wasmLetrasClassify", js.FuncOf(wasmLetrasClassify))
	g.Set("wasmLetrasDataset", js.FuncOf(wasmLetrasDataset))

	// MADALINE
	g.Set("wasmMadTrain", js.FuncOf(wasmMadTrain))
	g.Set("wasmMadResult", js.FuncOf(wasmMadResult))
	g.Set("wasmMadClassify", js.FuncOf(wasmMadClassify))
	g.Set("wasmMadDataset", js.FuncOf(wasmMadDataset))

	// Image Regression — Standard
	g.Set("wasmImgregTarget", js.FuncOf(wasmImgregTarget))
	g.Set("wasmImgregTrain", js.FuncOf(wasmImgregTrain))
	g.Set("wasmImgregReset", js.FuncOf(wasmImgregReset))

	// Image Regression — Goroutines
	g.Set("wasmIgorTrain", js.FuncOf(wasmIgorTrain))
	g.Set("wasmIgorReset", js.FuncOf(wasmIgorReset))

	// Image Regression — Minibatch
	g.Set("wasmImbTrain", js.FuncOf(wasmImbTrain))
	g.Set("wasmImbReset", js.FuncOf(wasmImbReset))

	// Signal ready
	g.Get("document").Call("dispatchEvent",
		js.Global().Get("CustomEvent").New("wasmReady"))

	// Block forever
	<-make(chan struct{})
}
