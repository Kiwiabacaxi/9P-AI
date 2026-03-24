package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

// =============================================================================
// Estado global — redes treinadas em memória
// =============================================================================

var (
	mu sync.RWMutex

	// MLP Desafio
	mlpRede *MLP
	mlpRes  *MLPResult

	// MLP Letras
	ltrRede     *LtrMLP
	ltrRes      *LtrResult
	ltrTraining bool

	// Hebb
	hebbRedes map[string]*HebbResult

	// Perceptron Portas
	percPortasRedes map[string]*PercPortasResult

	// Perceptron Letras
	percLetrasRede *PercLetrasResult

	// MADALINE
	madRede     *MadNet
	madRes      *MadResult
	madTraining bool
)

func init() {
	hebbRedes = make(map[string]*HebbResult)
	percPortasRedes = make(map[string]*PercPortasResult)
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
	}
	writeJSON(w, http.StatusOK, status{
		MLPTrained:     mlpRede != nil,
		LetrasTrained:  ltrRede != nil,
		LtrTraining:    ltrTraining,
		HebbCount:      len(hebbRedes),
		PercPortaCount: len(percPortasRedes),
		PercLetrasDone: percLetrasRede != nil,
		MadTrained:     madRede != nil,
		MadTraining:    madTraining,
	})
}

// =============================================================================
// MLP Desafio
// =============================================================================

func handleMLPTrain(w http.ResponseWriter, r *http.Request) {
	res := mlpTreinar()
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

	progressCh := make(chan LtrStep, 64)
	go func() {
		res, rede := ltrTreinar(progressCh)
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
	var req LtrClassifyReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, ltrClassificar(*rede, req.Grade))
}

func handleLetrasDataset(w http.ResponseWriter, r *http.Request) {
	dataset := ltrDataset()
	type entry struct {
		Letra string          `json:"letra"`
		Idx   int             `json:"idx"`
		Grade [ltrNIn]float64 `json:"grade"`
	}
	entries := make([]entry, ltrNOut)
	for i := 0; i < ltrNOut; i++ {
		entries[i] = entry{Letra: ltrNomes[i], Idx: i, Grade: dataset[i]}
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
	for _, p := range portasLogicas() {
		_, trained := hebbRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Desc: p.Desc, Trained: trained})
	}
	writeJSON(w, http.StatusOK, result)
}

// POST /api/hebb/train?porta=AND  (ou porta=ALL)
func handleHebbTrain(w http.ResponseWriter, r *http.Request) {
	nomePorta := r.URL.Query().Get("porta")
	portas := portasLogicas()

	if nomePorta == "ALL" || nomePorta == "" {
		results := make(map[string]*HebbResult)
		for _, p := range portas {
			res := hebbTreinar(p)
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

	var found *PortaLogica
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

	res := hebbTreinar(*found)
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
	for _, p := range portasLogicas() {
		_, trained := percPortasRedes[p.Nome]
		result = append(result, portaInfo{Nome: p.Nome, Trained: trained})
	}
	writeJSON(w, http.StatusOK, result)
}

// POST /api/perceptron-portas/train?porta=AND  (ou ALL)
func handlePercPortasTrain(w http.ResponseWriter, r *http.Request) {
	nomePorta := r.URL.Query().Get("porta")
	portas := portasLogicas()

	if nomePorta == "ALL" || nomePorta == "" {
		results := make(map[string]*PercPortasResult)
		for _, p := range portas {
			res := percPortasTreinar(p)
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

	var found *PortaLogica
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

	res := percPortasTreinar(*found)
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
	res := percLetrasTreinar()
	mu.Lock()
	percLetrasRede = &res
	mu.Unlock()
	writeJSON(w, http.StatusOK, res)
}

// GET /api/perceptron-letras/dataset
func handlePercLetrasDataset(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, []PercLetrasDatasetResp{
		{Letra: "A", Grade: percLetraA()},
		{Letra: "B", Grade: percLetraB()},
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

	progressCh := make(chan MadStep, 64)
	go func() {
		res, rede := madTreinar(progressCh)
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
	var req MadClassifyReq
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		errJSON(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, madClassificar(*rede, req.Grade))
}

// GET /api/madaline/dataset
func handleMadDataset(w http.ResponseWriter, r *http.Request) {
	dataset := madDataset()
	type entry struct {
		Letra string        `json:"letra"`
		Idx   int           `json:"idx"`
		Grade [madNIn]float64 `json:"grade"`
	}
	entries := make([]entry, madNLetras)
	for i := 0; i < madNLetras; i++ {
		var g [madNIn]float64
		for j := 0; j < madNIn; j++ {
			g[j] = float64(dataset[i][j])
		}
		entries[i] = entry{Letra: madNomes[i], Idx: i, Grade: g}
	}
	writeJSON(w, http.StatusOK, entries)
}

// =============================================================================
// main
// =============================================================================

func main() {
	mux := http.NewServeMux()

	mux.Handle("/", http.FileServer(http.Dir("../static")))

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

	addr := ":8080"
	log.Printf("MLP Web Server rodando em http://localhost%s", addr)
	log.Fatal(http.ListenAndServe(addr, mux))
}
