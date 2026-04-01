package timeseries

// Save/Load de modelos treinados — mesmo padrão do CNN

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

type ModelMeta struct {
	ID           string  `json:"id"`
	Nome         string  `json:"nome"`
	CriadoEm    string  `json:"criadoEm"`
	Ticker       string  `json:"ticker"`
	WindowSize   int     `json:"windowSize"`
	HiddenSize   int     `json:"hiddenSize"`
	Ciclos       int     `json:"ciclos"`
	RmseFinal    float64 `json:"rmseFinal"`
	MaeFinal     float64 `json:"maeFinal"`
	PredicaoAmanha float64 `json:"predicaoAmanha"`
}

func SaveModel(dir string, net *TimeSeriesMLP, result *TimeSeriesResult, nome string) (ModelMeta, error) {
	now := time.Now()
	id := now.Format("20060102-150405")
	modelDir := filepath.Join(dir, id)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return ModelMeta{}, err
	}

	// Pesos
	wf, err := os.Create(filepath.Join(modelDir, "weights.gob"))
	if err != nil { return ModelMeta{}, err }
	defer wf.Close()
	if err := gob.NewEncoder(wf).Encode(net); err != nil { return ModelMeta{}, err }

	// Resultado
	rd, _ := json.MarshalIndent(result, "", "  ")
	os.WriteFile(filepath.Join(modelDir, "result.json"), rd, 0644)

	// Meta
	meta := ModelMeta{
		ID: id, Nome: nome, CriadoEm: now.Format(time.RFC3339),
		Ticker: result.Ticker, WindowSize: net.InSize, HiddenSize: net.HidSize,
		Ciclos: result.Ciclos, RmseFinal: result.RmseFinal, MaeFinal: result.MaeFinal,
		PredicaoAmanha: result.PredicaoAmanha,
	}
	md, _ := json.MarshalIndent(meta, "", "  ")
	os.WriteFile(filepath.Join(modelDir, "meta.json"), md, 0644)

	return meta, nil
}

func LoadModel(dir, modelID string) (*TimeSeriesMLP, *TimeSeriesResult, error) {
	modelDir := filepath.Join(dir, modelID)
	wf, err := os.Open(filepath.Join(modelDir, "weights.gob"))
	if err != nil { return nil, nil, fmt.Errorf("modelo não encontrado: %w", err) }
	defer wf.Close()

	var net TimeSeriesMLP
	if err := gob.NewDecoder(wf).Decode(&net); err != nil { return nil, nil, err }

	rd, err := os.ReadFile(filepath.Join(modelDir, "result.json"))
	if err != nil { return nil, nil, err }
	var result TimeSeriesResult
	json.Unmarshal(rd, &result)

	return &net, &result, nil
}

func ListModels(dir string) ([]ModelMeta, error) {
	os.MkdirAll(dir, 0755)
	entries, err := os.ReadDir(dir)
	if err != nil { return nil, err }

	models := make([]ModelMeta, 0)
	for _, e := range entries {
		if !e.IsDir() { continue }
		data, err := os.ReadFile(filepath.Join(dir, e.Name(), "meta.json"))
		if err != nil { continue }
		var meta ModelMeta
		if json.Unmarshal(data, &meta) == nil {
			models = append(models, meta)
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ID > models[j].ID })
	return models, nil
}

func DeleteModel(dir, modelID string) error {
	modelDir := filepath.Join(dir, modelID)
	if _, err := os.Stat(filepath.Join(modelDir, "meta.json")); err != nil {
		return fmt.Errorf("modelo não encontrado: %s", modelID)
	}
	return os.RemoveAll(modelDir)
}
