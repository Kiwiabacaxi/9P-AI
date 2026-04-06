package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"

	"mlp-server/hebb"
	"mlp-server/imgreg"
	igoroutines "mlp-server/imgreg_goroutines"
	imatrix     "mlp-server/imgreg_matrix"
	iminibatch  "mlp-server/imgreg_minibatch"
	ibench      "mlp-server/imgreg_bench"
	"mlp-server/letras"
	"mlp-server/madaline"
	"mlp-server/mlp"
	"mlp-server/mlpfunc"
	"mlp-server/mlport"
	"mlp-server/cnn"
	"mlp-server/timeseries"
	perceptronletras "mlp-server/perceptron_letras"
	perceptronportas "mlp-server/perceptron_portas"
)

// =============================================================================
// Estado global — redes treinadas em memória
// =============================================================================

var (
	mu sync.RWMutex

	// MLP Desafio
	mlpRede *mlp.MLP
	mlpRes  *mlp.MLPResult

	// MLP Letras
	ltrRede     *letras.LtrMLP
	ltrRes      *letras.LtrResult
	ltrTraining bool

	// Hebb
	hebbRedes map[string]*hebb.HebbResult

	// Perceptron Portas
	percPortasRedes map[string]*perceptronportas.PercPortasResult

	// Perceptron Letras
	percLetrasRede *perceptronletras.PercLetrasResult

	// MADALINE
	madRede     *madaline.MadNet
	madRes      *madaline.MadResult
	madTraining bool

	// Image Regression
	imgregRede     *imgreg.Net
	imgregTraining bool
	imgregCfg      *imgreg.Config
	imgregCancel   context.CancelFunc // cancela treino em andamento

	// imgreg_goroutines
	igorRede     *igoroutines.Net
	igorTraining bool
	igorCfg      *igoroutines.Config
	igorCancel   context.CancelFunc

	// imgreg_matrix
	imatRede     *imatrix.Net
	imatTraining bool
	imatCfg      *imatrix.Config
	imatCancel   context.CancelFunc

	// imgreg_minibatch
	imbRede      *iminibatch.Net
	imbTraining  bool
	imbCfg       *iminibatch.Config
	imbCancel    context.CancelFunc

	// imgreg_bench
	benchCfg     *ibench.BenchConfig
	benchRunning bool
	benchCancel  context.CancelFunc

	// MLP Funcoes (aproximacao de funcao)
	mlpFuncCfg      *mlpfunc.Config
	mlpFuncRes      *mlpfunc.FuncResult
	mlpFuncTraining bool

	// MLP Ortogonal (vetores bipolares)
	ortCfg      *mlport.Config
	ortRede     *mlport.OrtMLP
	ortRes      *mlport.OrtResult
	ortTraining bool

	// CNN (EMNIST Letters)
	cnnRede     *cnn.CNN
	cnnRes      *cnn.CnnResult
	cnnCfg      *cnn.Config
	cnnTraining bool
	cnnCancel   context.CancelFunc
	cnnData     *cnn.EMNISTData

	// Time Series (Previsão de ações)
	tsRede      *timeseries.TimeSeriesMLP
	tsRes       *timeseries.TimeSeriesResult
	tsCfg       *timeseries.Config
	tsTraining  bool
	tsStockData *timeseries.StockData
	tsNormData  *timeseries.NormalizedData
)

func init() {
	hebbRedes = make(map[string]*hebb.HebbResult)
	percPortasRedes = make(map[string]*perceptronportas.PercPortasResult)
}

// =============================================================================
// Helpers HTTP
// =============================================================================

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func errJSON(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func cors(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next(w, r)
	}
}

// =============================================================================
// Status
// =============================================================================

func handleStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	defer mu.RUnlock()
	type status struct {
		MLPTrained     bool `json:"mlpTrained"`
		LetrasTrained  bool `json:"letrasTrained"`
		LtrTraining    bool `json:"ltrTraining"`
		HebbCount      int  `json:"hebbCount"`
		PercPortaCount int  `json:"percPortaCount"`
		PercLetrasDone bool `json:"percLetrasDone"`
		MadTrained     bool `json:"madTrained"`
		MadTraining    bool `json:"madTraining"`
		ImgregTrained   bool `json:"imgregTrained"`
		ImgregTraining  bool `json:"imgregTraining"`
		MlpFuncTrained  bool `json:"mlpFuncTrained"`
		MlpFuncTraining bool `json:"mlpFuncTraining"`
		OrtTrained      bool `json:"ortTrained"`
		OrtTraining     bool `json:"ortTraining"`
		CnnTrained      bool `json:"cnnTrained"`
		CnnTraining     bool `json:"cnnTraining"`
		TsTrained       bool `json:"tsTrained"`
		TsTraining      bool `json:"tsTraining"`
	}
	writeJSON(w, http.StatusOK, status{
		MLPTrained:      mlpRede != nil,
		LetrasTrained:   ltrRede != nil,
		LtrTraining:     ltrTraining,
		HebbCount:       len(hebbRedes),
		PercPortaCount:  len(percPortasRedes),
		PercLetrasDone:  percLetrasRede != nil,
		MadTrained:      madRede != nil,
		MadTraining:     madTraining,
		ImgregTrained:   imgregRede != nil,
		ImgregTraining:  imgregTraining,
		MlpFuncTrained:  mlpFuncRes != nil,
		MlpFuncTraining: mlpFuncTraining,
		OrtTrained:      ortRede != nil,
		OrtTraining:     ortTraining,
		CnnTrained:      cnnRede != nil,
		CnnTraining:     cnnTraining,
		TsTrained:       tsRede != nil,
		TsTraining:      tsTraining,
	})
}

// =============================================================================
// MLP Desafio
// =============================================================================

func handleMLPTrain(w http.ResponseWriter, r *http.Request) {
	res := mlp.Treinar()
	mu.Lock()
	mlpRes = &res
	rede := res.Rede
	mlpRede = &rede
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

func handleMLPResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	res := mlpRes
	mu.RUnlock()
	if res == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada — POST /api/mlp/train primeiro")
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// =============================================================================
// MLP Letras
// =============================================================================

func handleLetrasTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if ltrTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	ltrTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	progressCh := make(chan letras.LtrStep, 64)
	go func() {
		res, rede := letras.Treinar(progressCh)
		mu.Lock()
		ltrRes = &res
		ltrRede = &rede
		ltrTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := ltrRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

func handleLetrasResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	res := ltrRes
	mu.RUnlock()
	if res == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func handleLetrasClassify(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	rede := ltrRede
	mu.RUnlock()
	if rede == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	var req letras.ClassifyReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, letras.Classificar(*rede, req.Grade))
}

func handleLetrasDataset(w http.ResponseWriter, r *http.Request) {
	dataset := letras.Dataset()
	type entry struct {
		Letra string           `json:"letra"`
		Idx   int              `json:"idx"`
		Grade [letras.NIn]float64 `json:"grade"`
	}
	entries := make([]entry, letras.NOut)
	for i := 0; i < letras.NOut; i++ {
		entries[i] = entry{Letra: letras.Nomes[i], Idx: i, Grade: dataset[i]}
	}
	writeJSON(w, http.StatusOK, entries)
}

// =============================================================================
// Hebb
// =============================================================================

// GET /api/hebb/portas
func handleHebbPortas(w http.ResponseWriter, r *http.Request) {
	type portaInfo struct {
		Nome    string `json:"nome"`
		Desc    string `json:"desc"`
		Trained bool   `json:"trained"`
	}
	mu.RLock()
	defer mu.RUnlock()
	var result []portaInfo
	for _, p := range hebb.PortasLogicas() {
		_, trained := hebbRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Desc: p.Desc, Trained: trained})
	}
	writeJSON(w, http.StatusOK, result)
}

// POST /api/hebb/train?porta=AND  (ou porta=ALL)
func handleHebbTrain(w http.ResponseWriter, r *http.Request) {
	nomePorta := r.URL.Query().Get("porta")
	portas := hebb.PortasLogicas()

	if nomePorta == "ALL" || nomePorta == "" {
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
		writeJSON(w, http.StatusOK, results)
		return
	}

	var found *hebb.PortaLogica
	for _, p := range portas {
		if p.Nome == nomePorta {
			pp := p
			found = &pp
			break
		}
	}
	if found == nil {
		errJSON(w, http.StatusBadRequest, "porta inválida: "+nomePorta)
		return
	}

	res := hebb.Treinar(*found)
	mu.Lock()
	hebbRedes[found.Nome] = &res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

// =============================================================================
// Perceptron Portas
// =============================================================================

// GET /api/perceptron-portas/portas
func handlePercPortasLista(w http.ResponseWriter, r *http.Request) {
	type portaInfo struct {
		Nome    string `json:"nome"`
		Trained bool   `json:"trained"`
	}
	mu.RLock()
	defer mu.RUnlock()
	var result []portaInfo
	for _, p := range perceptronportas.PortasLogicas() {
		_, trained := percPortasRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Trained: trained})
	}
	writeJSON(w, http.StatusOK, result)
}

// POST /api/perceptron-portas/train?porta=AND  (ou ALL)
func handlePercPortasTrain(w http.ResponseWriter, r *http.Request) {
	nomePorta := r.URL.Query().Get("porta")
	portas := perceptronportas.PortasLogicas()

	if nomePorta == "ALL" || nomePorta == "" {
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
		writeJSON(w, http.StatusOK, results)
		return
	}

	var found *perceptronportas.PortaLogica
	for _, p := range portas {
		if p.Nome == nomePorta {
			pp := p
			found = &pp
			break
		}
	}
	if found == nil {
		errJSON(w, http.StatusBadRequest, "porta inválida: "+nomePorta)
		return
	}

	res := perceptronportas.Treinar(*found)
	mu.Lock()
	percPortasRedes[found.Nome] = &res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

// =============================================================================
// Perceptron Letras
// =============================================================================

// POST /api/perceptron-letras/train
func handlePercLetrasTrain(w http.ResponseWriter, r *http.Request) {
	res := perceptronletras.Treinar()
	mu.Lock()
	percLetrasRede = &res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

// GET /api/perceptron-letras/dataset
func handlePercLetrasDataset(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, []perceptronletras.DatasetResp{
		{Letra: "A", Grade: perceptronletras.LetraA()},
		{Letra: "B", Grade: perceptronletras.LetraB()},
	})
}

// =============================================================================
// MADALINE
// =============================================================================

// POST /api/madaline/train  (SSE)
func handleMadTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if madTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	madTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	progressCh := make(chan madaline.MadStep, 64)
	go func() {
		res, rede := madaline.Treinar(progressCh)
		mu.Lock()
		madRes = &res
		madRede = &rede
		madTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := madRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

// GET /api/madaline/result
func handleMadResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	res := madRes
	mu.RUnlock()
	if res == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	writeJSON(w, http.StatusOK, res)
}

// POST /api/madaline/classify
func handleMadClassify(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	rede := madRede
	mu.RUnlock()
	if rede == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	var req madaline.MadClassifyReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, madaline.Classificar(*rede, req.Grade))
}

// GET /api/madaline/dataset
func handleMadDataset(w http.ResponseWriter, r *http.Request) {
	dataset := madaline.Dataset()
	type entry struct {
		Letra string              `json:"letra"`
		Idx   int                 `json:"idx"`
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
	writeJSON(w, http.StatusOK, entries)
}

// =============================================================================
// Image Regression
// =============================================================================

// POST /api/imgreg/config
// Salva a configuração escolhida pelo usuário antes do treino.
// Separado do SSE para permitir GET no EventSource.
func handleImgregConfig(w http.ResponseWriter, r *http.Request) {
	var cfg imgreg.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	imgregCfg = &cfg
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

// GET /api/imgreg/train  (SSE via EventSource)
// Inicia o treinamento com a config previamente salva via POST /api/imgreg/config.
// Usa GET para compatibilidade com EventSource do browser.
func handleImgregTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imgregTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	cfg := imgregCfg
	if cfg == nil {
		mu.Unlock()
		errJSON(w, http.StatusBadRequest, "configure primeiro via POST /api/imgreg/config")
		return
	}
	imgregTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		mu.Lock()
		imgregTraining = false
		mu.Unlock()
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	// Canal com buffer generoso; o treinamento envia steps periódicos + 1 step final com Done=true
	ctx, cancel := context.WithCancel(r.Context())
	progressCh := make(chan imgreg.Step, 64)
	mu.Lock()
	imgregCancel = cancel
	mu.Unlock()
	go func() {
		defer cancel()
		rede := imgreg.Treinar(ctx, *cfg, progressCh)
		mu.Lock()
		imgregRede = &rede
		imgregTraining = false
		imgregCancel = nil
		mu.Unlock()
	}()

	// Lê todos os steps do canal. O último terá Done=true e é enviado como evento "done".
	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

// GET /api/imgreg/target?img=coracao
// Retorna os 256 pixels RGB [0,1] da imagem-alvo escolhida (sem treinar).
func handleImgregTarget(w http.ResponseWriter, r *http.Request) {
	img := r.URL.Query().Get("img")
	if img == "" {
		img = "coracao"
	}
	pixels := imgreg.GetTarget(img)
	writeJSON(w, http.StatusOK, pixels)
}

// POST /api/imgreg/reset
// Cancela qualquer treino em andamento e limpa o estado da rede.
func handleImgregReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imgregCancel != nil {
		imgregCancel()
		imgregCancel = nil
	}
	imgregRede = nil
	imgregCfg = nil
	imgregTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

// GET /api/imgreg/status
func handleImgregStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	training := imgregTraining
	trained := imgregRede != nil
	cfg := imgregCfg
	mu.RUnlock()

	type resp struct {
		Training bool   `json:"training"`
		Trained  bool   `json:"trained"`
		Imagem   string `json:"imagem"`
	}
	out := resp{Training: training, Trained: trained}
	if cfg != nil {
		out.Imagem = cfg.Imagem
	}
	writeJSON(w, http.StatusOK, out)
}

// =============================================================================
// Image Regression — Goroutines backend
// =============================================================================

func handleIgorConfig(w http.ResponseWriter, r *http.Request) {
	var cfg igoroutines.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); igorCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleIgorTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if igorTraining { mu.Unlock(); errJSON(w, http.StatusConflict, "treino em andamento"); return }
	cfg := igorCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	igorTraining = true
	ctx, cancel := context.WithCancel(r.Context())
	igorCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); cancel(); igorCancel = nil; igorTraining = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	progressCh := make(chan igoroutines.Step, 64)
	go func() {
		defer cancel()
		rede := igoroutines.Treinar(ctx, *cfg, progressCh)
		mu.Lock(); igorRede = &rede; igorTraining = false; igorCancel = nil; mu.Unlock()
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleIgorReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if igorCancel != nil { igorCancel(); igorCancel = nil }
	igorRede = nil; igorCfg = nil; igorTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleIgorStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	training, trained := igorTraining, igorRede != nil
	mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]any{"training": training, "trained": trained})
}

// =============================================================================
// Image Regression — Matrix backend
// =============================================================================

func handleImatConfig(w http.ResponseWriter, r *http.Request) {
	var cfg imatrix.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); imatCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleImatTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imatTraining { mu.Unlock(); errJSON(w, http.StatusConflict, "treino em andamento"); return }
	cfg := imatCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	imatTraining = true
	ctx, cancel := context.WithCancel(r.Context())
	imatCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); cancel(); imatCancel = nil; imatTraining = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	progressCh := make(chan imatrix.Step, 64)
	go func() {
		defer cancel()
		rede := imatrix.Treinar(ctx, *cfg, progressCh)
		mu.Lock(); imatRede = &rede; imatTraining = false; imatCancel = nil; mu.Unlock()
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleImatReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imatCancel != nil { imatCancel(); imatCancel = nil }
	imatRede = nil; imatCfg = nil; imatTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleImatStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	training, trained := imatTraining, imatRede != nil
	mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]any{"training": training, "trained": trained})
}

// =============================================================================
// Image Regression — Mini-batch backend
// =============================================================================

func handleImbConfig(w http.ResponseWriter, r *http.Request) {
	var cfg iminibatch.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); imbCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleImbTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imbTraining { mu.Unlock(); errJSON(w, http.StatusConflict, "treino em andamento"); return }
	cfg := imbCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	imbTraining = true
	ctx, cancel := context.WithCancel(r.Context())
	imbCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); cancel(); imbCancel = nil; imbTraining = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	progressCh := make(chan iminibatch.Step, 64)
	go func() {
		defer cancel()
		rede := iminibatch.Treinar(ctx, *cfg, progressCh)
		mu.Lock(); imbRede = &rede; imbTraining = false; imbCancel = nil; mu.Unlock()
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleImbReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if imbCancel != nil { imbCancel(); imbCancel = nil }
	imbRede = nil; imbCfg = nil; imbTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleImbStatus(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	training, trained := imbTraining, imbRede != nil
	mu.RUnlock()
	writeJSON(w, http.StatusOK, map[string]any{"training": training, "trained": trained})
}

// =============================================================================
// Image Regression — Benchmark
// =============================================================================

func handleBenchConfig(w http.ResponseWriter, r *http.Request) {
	var cfg ibench.BenchConfig
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error()); return
	}
	mu.Lock(); benchCfg = &cfg; mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleBenchTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if benchRunning { mu.Unlock(); errJSON(w, http.StatusConflict, "benchmark em andamento"); return }
	cfg := benchCfg
	if cfg == nil { mu.Unlock(); errJSON(w, http.StatusBadRequest, "configure primeiro"); return }
	benchRunning = true
	ctx, cancel := context.WithCancel(r.Context())
	benchCancel = cancel
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok { mu.Lock(); cancel(); benchCancel = nil; benchRunning = false; mu.Unlock(); errJSON(w, 500, "streaming não suportado"); return }

	benchCh := make(chan ibench.BenchStep, 192)
	go func() {
		defer cancel()
		ibench.Rodar(ctx, *cfg, benchCh)
		mu.Lock(); benchRunning = false; benchCancel = nil; mu.Unlock()
	}()

	for step := range benchCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}
}

func handleBenchReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if benchCancel != nil { benchCancel(); benchCancel = nil }
	benchCfg = nil; benchRunning = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

// =============================================================================
// MLP Funcoes (aproximacao de funcao)
// =============================================================================

func handleMlpFuncConfig(w http.ResponseWriter, r *http.Request) {
	var cfg mlpfunc.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	mlpFuncCfg = &cfg
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleMlpFuncTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if mlpFuncTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	cfg := mlpFuncCfg
	if cfg == nil {
		def := mlpfunc.DefaultConfig()
		cfg = &def
	}
	mlpFuncTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		mu.Lock()
		mlpFuncTraining = false
		mu.Unlock()
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	useCfg := *cfg
	progressCh := make(chan mlpfunc.FuncStep, 64)
	go func() {
		res := mlpfunc.Treinar(progressCh, useCfg)
		mu.Lock()
		mlpFuncRes = &res
		mlpFuncTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := mlpFuncRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

func handleMlpFuncReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	mlpFuncRes = nil
	mlpFuncCfg = nil
	mlpFuncTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleMlpFuncResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	res := mlpFuncRes
	mu.RUnlock()
	if res == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func handleMlpFuncFuncoes(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, mlpfunc.FuncoesDisponiveis())
}

// =============================================================================
// MLP Ortogonal (vetores bipolares)
// =============================================================================

func handleOrtConfig(w http.ResponseWriter, r *http.Request) {
	var cfg mlport.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	mu.Lock()
	ortCfg = &cfg
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "config salva"})
}

func handleOrtTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if ortTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	cfg := ortCfg
	if cfg == nil {
		def := mlport.DefaultConfig()
		cfg = &def
	}
	ortTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		mu.Lock()
		ortTraining = false
		mu.Unlock()
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	useCfg := *cfg
	progressCh := make(chan mlport.OrtStep, 64)
	go func() {
		res, rede := mlport.Treinar(progressCh, useCfg)
		mu.Lock()
		ortRes = &res
		ortRede = &rede
		ortTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := ortRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

func handleOrtReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	ortRede = nil
	ortRes = nil
	ortCfg = nil
	ortTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"ok": "resetado"})
}

func handleOrtResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	res := ortRes
	mu.RUnlock()
	if res == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	writeJSON(w, http.StatusOK, res)
}

func handleOrtClassify(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	rede := ortRede
	mu.RUnlock()
	if rede == nil {
		errJSON(w, http.StatusNotFound, "rede não treinada")
		return
	}
	var req mlport.ClassifyReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, mlport.Classificar(*rede, req.Grade))
}

func handleOrtDataset(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, mlport.GetDatasetInfo())
}

// =============================================================================
// main
// =============================================================================

// =============================================================================
// CNN (EMNIST Letters)
// =============================================================================

func handleCnnConfig(w http.ResponseWriter, r *http.Request) {
	var cfg cnn.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, "config inválida: "+err.Error())
		return
	}
	mu.Lock()
	cnnCfg = &cfg
	mu.Unlock()
	writeJSON(w, http.StatusOK, cfg)
}

func handleCnnTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if cnnTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	cfg := cnnCfg
	if cfg == nil {
		def := cnn.DefaultConfig()
		cfg = &def
	}

	// Carregar dataset se ainda não carregado
	if cnnData == nil {
		data, err := cnn.LoadEMNIST("data/emnist")
		if err != nil {
			mu.Unlock()
			errJSON(w, http.StatusInternalServerError, "erro ao carregar EMNIST: "+err.Error())
			return
		}
		cnnData = data
	}

	cnnTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		mu.Lock()
		cnnTraining = false
		mu.Unlock()
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	useCfg := *cfg
	ctx, cancel := context.WithCancel(context.Background())

	mu.Lock()
	cnnCancel = cancel
	mu.Unlock()

	progressCh := make(chan cnn.CnnStep, 64)
	go func() {
		net, res := cnn.Treinar(ctx, useCfg, cnnData, progressCh)
		mu.Lock()
		cnnRede = net
		cnnRes = &res
		cnnTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := cnnRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

func handleCnnReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if cnnCancel != nil {
		cnnCancel()
	}
	cnnRede = nil
	cnnRes = nil
	cnnCfg = nil
	cnnTraining = false
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}

func handleCnnResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	defer mu.RUnlock()
	if cnnRes == nil {
		errJSON(w, http.StatusNotFound, "CNN não treinada")
		return
	}
	writeJSON(w, http.StatusOK, cnnRes)
}

func handleCnnClassify(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Pixels []float64 `json:"pixels"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	mu.RLock()
	net := cnnRede
	mu.RUnlock()
	if net == nil {
		errJSON(w, http.StatusPreconditionFailed, "CNN não treinada")
		return
	}
	resp := cnn.Classificar(net, req.Pixels)
	writeJSON(w, http.StatusOK, resp)
}

const cnnModelsDir = "data/cnn-models"

func handleCnnVisualize(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Pixels []float64 `json:"pixels"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	mu.RLock()
	net := cnnRede
	mu.RUnlock()
	if net == nil {
		errJSON(w, http.StatusPreconditionFailed, "CNN não treinada")
		return
	}
	resp := cnn.Visualizar(net, req.Pixels)
	writeJSON(w, http.StatusOK, resp)
}

func handleCnnSave(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Nome string `json:"nome"`
	}
	json.NewDecoder(r.Body).Decode(&req)
	mu.RLock()
	net := cnnRede
	res := cnnRes
	mu.RUnlock()
	if net == nil || res == nil {
		errJSON(w, http.StatusPreconditionFailed, "CNN não treinada")
		return
	}
	meta, err := cnn.SaveModel(cnnModelsDir, net, res, req.Nome)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, "erro ao salvar: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, meta)
}

func handleCnnModels(w http.ResponseWriter, r *http.Request) {
	models, err := cnn.ListModels(cnnModelsDir)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, "erro ao listar: "+err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func handleCnnLoad(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	net, res, err := cnn.LoadModel(cnnModelsDir, req.ID)
	if err != nil {
		errJSON(w, http.StatusNotFound, "erro ao carregar: "+err.Error())
		return
	}
	mu.Lock()
	cnnRede = net
	cnnRes = res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

func handleCnnDeleteModel(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ID string `json:"id"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	if err := cnn.DeleteModel(cnnModelsDir, req.ID); err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

// =============================================================================
// Time Series (Previsão de ações)
// =============================================================================

const tsModelsDir = "data/ts-models"

func handleTsFetchData(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Ticker string `json:"ticker"`
		Period string `json:"period"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	if req.Ticker == "" { req.Ticker = "COGN3.SA" }
	if req.Period == "" { req.Period = "6mo" }

	data, err := timeseries.FetchStockData(req.Ticker, req.Period)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, "erro ao buscar dados: "+err.Error())
		return
	}
	mu.Lock()
	tsStockData = data
	mu.Unlock()
	writeJSON(w, http.StatusOK, data)
}

func handleTsConfig(w http.ResponseWriter, r *http.Request) {
	var cfg timeseries.Config
	if err := json.NewDecoder(r.Body).Decode(&cfg); err != nil {
		errJSON(w, http.StatusBadRequest, "config inválida")
		return
	}
	mu.Lock()
	tsCfg = &cfg
	mu.Unlock()
	writeJSON(w, http.StatusOK, cfg)
}

func handleTsTrain(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	if tsTraining {
		mu.Unlock()
		errJSON(w, http.StatusConflict, "treinamento já em andamento")
		return
	}
	cfg := tsCfg
	if cfg == nil {
		def := timeseries.DefaultConfig()
		cfg = &def
	}
	stockData := tsStockData
	if stockData == nil || len(stockData.Close) == 0 {
		mu.Unlock()
		errJSON(w, http.StatusPreconditionFailed, "busque dados primeiro (fetch-data)")
		return
	}
	tsTraining = true
	mu.Unlock()

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		mu.Lock()
		tsTraining = false
		mu.Unlock()
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	useCfg := *cfg
	normData := timeseries.PrepareData(stockData.Close, stockData.Dates, useCfg.WindowSize, useCfg.ValidDays, useCfg.ValidPct)

	progressCh := make(chan timeseries.TimeSeriesStep, 64)
	go func() {
		net, res := timeseries.Treinar(useCfg, normData, progressCh)
		mu.Lock()
		tsRede = net
		tsRes = &res
		tsNormData = &normData
		tsTraining = false
		mu.Unlock()
		close(progressCh)
	}()

	for step := range progressCh {
		data, _ := json.Marshal(step)
		fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
	}

	mu.RLock()
	finalRes := tsRes
	mu.RUnlock()
	if finalRes != nil {
		data, _ := json.Marshal(finalRes)
		fmt.Fprintf(w, "event: done\ndata: %s\n\n", data)
		flusher.Flush()
	}
}

func handleTsReset(w http.ResponseWriter, r *http.Request) {
	mu.Lock()
	tsRede = nil
	tsRes = nil
	tsCfg = nil
	tsTraining = false
	tsStockData = nil
	tsNormData = nil
	mu.Unlock()
	writeJSON(w, http.StatusOK, map[string]string{"status": "reset"})
}

func handleTsResult(w http.ResponseWriter, r *http.Request) {
	mu.RLock()
	defer mu.RUnlock()
	if tsRes == nil {
		errJSON(w, http.StatusNotFound, "não treinado")
		return
	}
	writeJSON(w, http.StatusOK, tsRes)
}

func handleTsSave(w http.ResponseWriter, r *http.Request) {
	var req struct { Nome string `json:"nome"` }
	json.NewDecoder(r.Body).Decode(&req)
	mu.RLock()
	net, res := tsRede, tsRes
	mu.RUnlock()
	if net == nil || res == nil {
		errJSON(w, http.StatusPreconditionFailed, "não treinado")
		return
	}
	meta, err := timeseries.SaveModel(tsModelsDir, net, res, req.Nome)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, meta)
}

func handleTsModels(w http.ResponseWriter, r *http.Request) {
	models, err := timeseries.ListModels(tsModelsDir)
	if err != nil {
		errJSON(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, models)
}

func handleTsLoad(w http.ResponseWriter, r *http.Request) {
	var req struct { ID string `json:"id"` }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	net, res, err := timeseries.LoadModel(tsModelsDir, req.ID)
	if err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	mu.Lock()
	tsRede = net
	tsRes = res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

func handleTsCompare(w http.ResponseWriter, r *http.Request) {
	var req struct {
		Modelos []string `json:"modelos"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}

	mu.RLock()
	stockData := tsStockData
	cfg := tsCfg
	mu.RUnlock()

	if stockData == nil || len(stockData.Close) == 0 {
		errJSON(w, http.StatusPreconditionFailed, "busque dados primeiro (fetch-data)")
		return
	}
	if cfg == nil {
		def := timeseries.DefaultConfig()
		cfg = &def
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	flusher, ok := w.(http.Flusher)
	if !ok {
		errJSON(w, http.StatusInternalServerError, "streaming não suportado")
		return
	}

	useCfg := *cfg
	normData := timeseries.PrepareData(stockData.Close, stockData.Dates, useCfg.WindowSize, useCfg.ValidDays, useCfg.ValidPct)

	sendModelDone := func(modelo string, result timeseries.TimeSeriesResult, erro string) {
		mr := timeseries.CompareModelResult{
			Modelo: modelo,
			Result: result,
			Cor:    timeseries.ModelColors[modelo],
			Erro:   erro,
		}
		data, _ := json.Marshal(mr)
		fmt.Fprintf(w, "event: model-done\ndata: %s\n\n", data)
		flusher.Flush()
	}

	sendProgress := func(modelo string, step timeseries.TimeSeriesStep) {
		type progMsg struct {
			Modelo string `json:"modelo"`
			timeseries.TimeSeriesStep
		}
		data, _ := json.Marshal(progMsg{Modelo: modelo, TimeSeriesStep: step})
		fmt.Fprintf(w, "event: progress\ndata: %s\n\n", data)
		flusher.Flush()
	}

	for _, modelo := range req.Modelos {
		// Notificar que o modelo está iniciando
		fmt.Fprintf(w, "event: model-start\ndata: {\"modelo\":%q}\n\n", modelo)
		flusher.Flush()

		switch modelo {
		case "SMA":
			res := timeseries.TreinarSMA(useCfg, normData)
			sendModelDone("SMA", res, "")
		case "EMA":
			res := timeseries.TreinarEMA(useCfg, normData)
			sendModelDone("EMA", res, "")
		case "ARIMA":
			res := timeseries.TreinarARIMA(useCfg, normData)
			sendModelDone("ARIMA", res, "")
		case "ProphetLike":
			res := timeseries.TreinarProphetLike(useCfg, normData)
			sendModelDone("ProphetLike", res, "")
		case "MLP":
			ch := make(chan timeseries.TimeSeriesStep, 64)
			go func() {
				_, res := timeseries.Treinar(useCfg, normData, ch)
				close(ch)
				sendModelDone("MLP", res, "")
			}()
			for step := range ch { sendProgress("MLP", step) }
		case "LSTM":
			ch := make(chan timeseries.TimeSeriesStep, 64)
			go func() {
				_, res := timeseries.TreinarLSTM(useCfg, normData, ch)
				close(ch)
				sendModelDone("LSTM", res, "")
			}()
			for step := range ch { sendProgress("LSTM", step) }
		case "GRU":
			ch := make(chan timeseries.TimeSeriesStep, 64)
			go func() {
				_, res := timeseries.TreinarGRU(useCfg, normData, ch)
				close(ch)
				sendModelDone("GRU", res, "")
			}()
			for step := range ch { sendProgress("GRU", step) }
		case "BiLSTM":
			ch := make(chan timeseries.TimeSeriesStep, 64)
			go func() {
				_, res := timeseries.TreinarBiLSTM(useCfg, normData, ch)
				close(ch)
				sendModelDone("BiLSTM", res, "")
			}()
			for step := range ch { sendProgress("BiLSTM", step) }
		case "Seq2Seq":
			ch := make(chan timeseries.TimeSeriesStep, 64)
			go func() {
				_, res := timeseries.TreinarSeq2Seq(useCfg, normData, ch)
				close(ch)
				sendModelDone("Seq2Seq", res, "")
			}()
			for step := range ch { sendProgress("Seq2Seq", step) }
		case "RandomForest", "XGBoost", "Prophet":
			res, err := timeseries.TreinarPythonModel(modelo, useCfg, stockData)
			if err != nil {
				sendModelDone(modelo, timeseries.TimeSeriesResult{}, err.Error())
			} else {
				sendModelDone(modelo, res, "")
			}
		default:
			sendModelDone(modelo, timeseries.TimeSeriesResult{}, "modelo desconhecido")
		}
	}

	fmt.Fprintf(w, "event: done\ndata: {\"status\":\"complete\"}\n\n")
	flusher.Flush()
}

func handleTsAvailableModels(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, timeseries.AvailableModels)
}

func handleTsDeleteModel(w http.ResponseWriter, r *http.Request) {
	var req struct { ID string `json:"id"` }
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, "request inválido")
		return
	}
	if err := timeseries.DeleteModel(tsModelsDir, req.ID); err != nil {
		errJSON(w, http.StatusNotFound, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted"})
}

func main() {
	mux := http.NewServeMux()

	fs := http.FileServer(http.Dir("../static"))
	mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("Pragma", "no-cache")
		w.Header().Set("Expires", "0")
		fs.ServeHTTP(w, r)
	}))

	mux.HandleFunc("/api/status", cors(handleStatus))

	// MLP Desafio
	mux.HandleFunc("/api/mlp/train", cors(handleMLPTrain))
	mux.HandleFunc("/api/mlp/result", cors(handleMLPResult))

	// MLP Letras
	mux.HandleFunc("/api/letras/train", cors(handleLetrasTrain))
	mux.HandleFunc("/api/letras/result", cors(handleLetrasResult))
	mux.HandleFunc("/api/letras/classify", cors(handleLetrasClassify))
	mux.HandleFunc("/api/letras/dataset", cors(handleLetrasDataset))

	// Hebb
	mux.HandleFunc("/api/hebb/portas", cors(handleHebbPortas))
	mux.HandleFunc("/api/hebb/train", cors(handleHebbTrain))

	// Perceptron Portas
	mux.HandleFunc("/api/perceptron-portas/portas", cors(handlePercPortasLista))
	mux.HandleFunc("/api/perceptron-portas/train", cors(handlePercPortasTrain))

	// Perceptron Letras
	mux.HandleFunc("/api/perceptron-letras/train", cors(handlePercLetrasTrain))
	mux.HandleFunc("/api/perceptron-letras/dataset", cors(handlePercLetrasDataset))

	// MADALINE
	mux.HandleFunc("/api/madaline/train", cors(handleMadTrain))
	mux.HandleFunc("/api/madaline/result", cors(handleMadResult))
	mux.HandleFunc("/api/madaline/classify", cors(handleMadClassify))
	mux.HandleFunc("/api/madaline/dataset", cors(handleMadDataset))

	// Image Regression
	mux.HandleFunc("/api/imgreg/config", cors(handleImgregConfig))
	mux.HandleFunc("/api/imgreg/train", cors(handleImgregTrain))
	mux.HandleFunc("/api/imgreg/target", cors(handleImgregTarget))
	mux.HandleFunc("/api/imgreg/reset", cors(handleImgregReset))
	mux.HandleFunc("/api/imgreg/status", cors(handleImgregStatus))

	// imgreg_goroutines
	mux.HandleFunc("/api/imgreg-goroutines/config", cors(handleIgorConfig))
	mux.HandleFunc("/api/imgreg-goroutines/train",  cors(handleIgorTrain))
	mux.HandleFunc("/api/imgreg-goroutines/reset",  cors(handleIgorReset))
	mux.HandleFunc("/api/imgreg-goroutines/status", cors(handleIgorStatus))

	// imgreg_matrix
	mux.HandleFunc("/api/imgreg-matrix/config", cors(handleImatConfig))
	mux.HandleFunc("/api/imgreg-matrix/train",  cors(handleImatTrain))
	mux.HandleFunc("/api/imgreg-matrix/reset",  cors(handleImatReset))
	mux.HandleFunc("/api/imgreg-matrix/status", cors(handleImatStatus))

	// imgreg_minibatch
	mux.HandleFunc("/api/imgreg-minibatch/config", cors(handleImbConfig))
	mux.HandleFunc("/api/imgreg-minibatch/train",  cors(handleImbTrain))
	mux.HandleFunc("/api/imgreg-minibatch/reset",  cors(handleImbReset))
	mux.HandleFunc("/api/imgreg-minibatch/status", cors(handleImbStatus))

	// benchmark
	mux.HandleFunc("/api/imgreg-bench/config", cors(handleBenchConfig))
	mux.HandleFunc("/api/imgreg-bench/train",  cors(handleBenchTrain))
	mux.HandleFunc("/api/imgreg-bench/reset",  cors(handleBenchReset))

	// MLP Funcoes (aproximacao de funcao)
	mux.HandleFunc("/api/mlpfunc/config", cors(handleMlpFuncConfig))
	mux.HandleFunc("/api/mlpfunc/train", cors(handleMlpFuncTrain))
	mux.HandleFunc("/api/mlpfunc/reset", cors(handleMlpFuncReset))
	mux.HandleFunc("/api/mlpfunc/result", cors(handleMlpFuncResult))
	mux.HandleFunc("/api/mlpfunc/funcoes", cors(handleMlpFuncFuncoes))

	// MLP Ortogonal (vetores bipolares)
	mux.HandleFunc("/api/mlport/config", cors(handleOrtConfig))
	mux.HandleFunc("/api/mlport/train", cors(handleOrtTrain))
	mux.HandleFunc("/api/mlport/reset", cors(handleOrtReset))
	mux.HandleFunc("/api/mlport/result", cors(handleOrtResult))
	mux.HandleFunc("/api/mlport/classify", cors(handleOrtClassify))
	mux.HandleFunc("/api/mlport/dataset", cors(handleOrtDataset))

	// CNN (EMNIST Letters)
	mux.HandleFunc("/api/cnn/config", cors(handleCnnConfig))
	mux.HandleFunc("/api/cnn/train", cors(handleCnnTrain))
	mux.HandleFunc("/api/cnn/reset", cors(handleCnnReset))
	mux.HandleFunc("/api/cnn/result", cors(handleCnnResult))
	mux.HandleFunc("/api/cnn/classify", cors(handleCnnClassify))
	mux.HandleFunc("/api/cnn/visualize", cors(handleCnnVisualize))
	mux.HandleFunc("/api/cnn/save", cors(handleCnnSave))
	mux.HandleFunc("/api/cnn/models", cors(handleCnnModels))
	mux.HandleFunc("/api/cnn/load", cors(handleCnnLoad))
	mux.HandleFunc("/api/cnn/delete-model", cors(handleCnnDeleteModel))

	// Time Series (Previsão de ações)
	mux.HandleFunc("/api/timeseries/fetch-data", cors(handleTsFetchData))
	mux.HandleFunc("/api/timeseries/config", cors(handleTsConfig))
	mux.HandleFunc("/api/timeseries/train", cors(handleTsTrain))
	mux.HandleFunc("/api/timeseries/reset", cors(handleTsReset))
	mux.HandleFunc("/api/timeseries/result", cors(handleTsResult))
	mux.HandleFunc("/api/timeseries/save", cors(handleTsSave))
	mux.HandleFunc("/api/timeseries/models", cors(handleTsModels))
	mux.HandleFunc("/api/timeseries/load", cors(handleTsLoad))
	mux.HandleFunc("/api/timeseries/delete-model", cors(handleTsDeleteModel))
	mux.HandleFunc("/api/timeseries/compare", cors(handleTsCompare))
	mux.HandleFunc("/api/timeseries/available-models", cors(handleTsAvailableModels))

	addr := ":8080"
	log.Printf("MLP Web Server rodando em http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
