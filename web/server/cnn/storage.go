package cnn

// =============================================================================
// Persistência de modelos CNN — Save/Load/List/Delete
//
// Modelos treinados são salvos em disco para evitar retreinar toda vez.
// Usa encoding/gob para serializar pesos e encoding/json para metadados.
//
// Estrutura:
//   data/cnn-models/
//     ├── 20260331-194500/
//     │   ├── weights.gob   — pesos da rede (CNN struct)
//     │   ├── result.json   — resultado do treinamento (CnnResult)
//     │   └── meta.json     — metadados (nome, data, acurácia)
//     └── 20260401-103000/
//         └── ...
// =============================================================================

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"time"
)

// ModelMeta contém informações sobre um modelo salvo
type ModelMeta struct {
	ID           string  `json:"id"`           // timestamp-based ID (ex: 20260331-194500)
	Nome         string  `json:"nome"`         // nome dado pelo usuário
	CriadoEm    string  `json:"criadoEm"`     // ISO 8601
	Epocas       int     `json:"epocas"`
	TrainLimit   int     `json:"trainLimit"`
	Acuracia     float64 `json:"acuracia"`     // treino
	AcuraciaTest float64 `json:"acuraciaTest"` // teste
	LossFinal    float64 `json:"lossFinal"`
}

// SaveModel salva a rede treinada e seu resultado em disco.
// Retorna os metadados do modelo salvo.
func SaveModel(dir string, net *CNN, result *CnnResult, nome string) (ModelMeta, error) {
	// Gerar ID baseado em timestamp
	now := time.Now()
	id := now.Format("20060102-150405")

	modelDir := filepath.Join(dir, id)
	if err := os.MkdirAll(modelDir, 0755); err != nil {
		return ModelMeta{}, fmt.Errorf("erro ao criar diretório: %w", err)
	}

	// Salvar pesos com gob
	weightsPath := filepath.Join(modelDir, "weights.gob")
	wf, err := os.Create(weightsPath)
	if err != nil {
		return ModelMeta{}, fmt.Errorf("erro ao criar weights.gob: %w", err)
	}
	defer wf.Close()
	if err := gob.NewEncoder(wf).Encode(net); err != nil {
		return ModelMeta{}, fmt.Errorf("erro ao serializar pesos: %w", err)
	}

	// Salvar resultado como JSON
	resultPath := filepath.Join(modelDir, "result.json")
	resultData, _ := json.MarshalIndent(result, "", "  ")
	if err := os.WriteFile(resultPath, resultData, 0644); err != nil {
		return ModelMeta{}, fmt.Errorf("erro ao salvar result.json: %w", err)
	}

	// Criar e salvar metadados
	meta := ModelMeta{
		ID:           id,
		Nome:         nome,
		CriadoEm:    now.Format(time.RFC3339),
		Epocas:       result.Epocas,
		Acuracia:     result.Acuracia,
		AcuraciaTest: result.AcuraciaTest,
		LossFinal:    result.LossFinal,
	}
	metaPath := filepath.Join(modelDir, "meta.json")
	metaData, _ := json.MarshalIndent(meta, "", "  ")
	if err := os.WriteFile(metaPath, metaData, 0644); err != nil {
		return ModelMeta{}, fmt.Errorf("erro ao salvar meta.json: %w", err)
	}

	return meta, nil
}

// LoadModel carrega um modelo salvo pelo ID.
// Retorna a rede e o resultado do treinamento.
func LoadModel(dir, modelID string) (*CNN, *CnnResult, error) {
	modelDir := filepath.Join(dir, modelID)

	// Carregar pesos
	weightsPath := filepath.Join(modelDir, "weights.gob")
	wf, err := os.Open(weightsPath)
	if err != nil {
		return nil, nil, fmt.Errorf("modelo não encontrado: %w", err)
	}
	defer wf.Close()

	var net CNN
	if err := gob.NewDecoder(wf).Decode(&net); err != nil {
		return nil, nil, fmt.Errorf("erro ao deserializar pesos: %w", err)
	}

	// Carregar resultado
	resultPath := filepath.Join(modelDir, "result.json")
	resultData, err := os.ReadFile(resultPath)
	if err != nil {
		return nil, nil, fmt.Errorf("erro ao ler result.json: %w", err)
	}
	var result CnnResult
	if err := json.Unmarshal(resultData, &result); err != nil {
		return nil, nil, fmt.Errorf("erro ao parsear result.json: %w", err)
	}

	return &net, &result, nil
}

// ListModels lista todos os modelos salvos no diretório, ordenados por data (mais recente primeiro).
func ListModels(dir string) ([]ModelMeta, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	models := make([]ModelMeta, 0)
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		metaPath := filepath.Join(dir, entry.Name(), "meta.json")
		data, err := os.ReadFile(metaPath)
		if err != nil {
			continue // pular diretórios sem meta.json
		}
		var meta ModelMeta
		if err := json.Unmarshal(data, &meta); err != nil {
			continue
		}
		models = append(models, meta)
	}

	// Ordenar por ID decrescente (mais recente primeiro)
	sort.Slice(models, func(i, j int) bool {
		return models[i].ID > models[j].ID
	})

	return models, nil
}

// DeleteModel remove um modelo salvo pelo ID.
func DeleteModel(dir, modelID string) error {
	modelDir := filepath.Join(dir, modelID)
	// Verificar que existe
	if _, err := os.Stat(filepath.Join(modelDir, "meta.json")); err != nil {
		return fmt.Errorf("modelo não encontrado: %s", modelID)
	}
	return os.RemoveAll(modelDir)
}
