package mlport

import "math"

// =============================================================================
// MLP com Vetores Bipolares Ortogonais — Classificação A-Z (Aula 06)
//
// Baseado na Aula 06 (Manzan 2016, Fausett 1994):
//   "Não usaremos o limiar... usaremos a distância Euclidiana
//    para identificar cada elemento."
//
// Diferença fundamental em relação ao MLP Letras:
//   - MLP Letras: saída one-hot (26 neurônios, 1 por letra)
//   - MLP Ortogonal: saída = vetor bipolar ortogonal de 32 dimensões
//     → A rede aprende a MAPEAR cada letra para seu vetor ortogonal
//     → A classificação usa DISTÂNCIA EUCLIDIANA entre a saída da rede
//       e cada vetor-alvo, escolhendo o de MENOR distância
//
// Fórmula da distância euclidiana (do slide):
//   D = sqrt( Σ_k (t_k - y_k)² )
//
// "Ao reconhecer um padrão, usamos a MENOR distância euclidiana"
//
// Arquitetura: 35 entradas (grade 5×7) → nHid ocultos (tanh) → 32 saídas (tanh)
// =============================================================================

// Config permite customizar hiperparâmetros via frontend
type Config struct {
	NHid     int     `json:"nHid"`     // neurônios na camada oculta
	Alfa     float64 `json:"alfa"`     // taxa de aprendizado
	MaxCiclo int     `json:"maxCiclo"` // máximo de épocas
}

// DefaultConfig retorna a configuração padrão
func DefaultConfig() Config {
	return Config{
		NHid:     15,
		Alfa:     0.01,
		MaxCiclo: 50000,
	}
}

// OrtMLP armazena os pesos da rede.
//
// Arquitetura de 2 camadas de pesos:
//   V[i][j] = peso entrada i → oculto j     (35 × nHid)
//   V0[j]   = bias do neurônio oculto j
//   W[j][k] = peso oculto j → saída k       (nHid × 32)
//   W0[k]   = bias do neurônio de saída k
type OrtMLP struct {
	nHid int
	V    [][]float64 // pesos entrada→oculta  [NIn][nHid]
	V0   []float64   // bias camada oculta    [nHid]
	W    [][]float64 // pesos oculta→saída    [nHid][NOrt]
	W0   []float64   // bias camada de saída  [NOrt]
}

// OrtStep é enviado via SSE durante o treinamento
type OrtStep struct {
	Ciclo       int     `json:"ciclo"`
	LetraIdx    int     `json:"letraIdx"`
	Letra       string  `json:"letra"`
	ErroTotal   float64 `json:"erroTotal"`
	ActiveLayer int     `json:"activeLayer"` // para animação do NetworkViz
}

// OrtResult é o resultado final do treinamento
type OrtResult struct {
	Convergiu     bool                `json:"convergiu"`
	Ciclos        int                 `json:"ciclos"`
	ErroFinal     float64             `json:"erroFinal"`
	ErroHistorico []float64           `json:"erroHistorico"`
	Acertos       int                 `json:"acertos"`
	Total         int                 `json:"total"`
	Acuracia      float64             `json:"acuracia"`
	Vetores       [NOrt][NOrt]float64 `json:"vetores"`
}

type ClassifyReq struct {
	Grade [NIn]float64 `json:"grade"`
}

type OrtCandidate struct {
	Letra     string  `json:"letra"`
	Distancia float64 `json:"distancia"`
	Idx       int     `json:"idx"`
}

type ClassifyResp struct {
	LetraIdx   int               `json:"letraIdx"`
	Letra      string            `json:"letra"`
	Distancias [NClasses]float64 `json:"distancias"`
	Top5       []OrtCandidate    `json:"top5"`
	SaidaRede  []float64         `json:"saidaRede"` // saída bruta da rede (32 valores)
}

// =============================================================================
// Distância Euclidiana (do slide)
//
// D = sqrt( Σ_k (t_k - y_k)² )
//
// Mede o "quão longe" a saída da rede (y) está do vetor-alvo (t).
// A letra classificada é aquela cujo vetor-alvo tem MENOR distância.
// =============================================================================

func distanciaEuclidiana(y []float64, t [NOrt]float64) float64 {
	var soma float64
	for k := range NOrt {
		d := t[k] - y[k]
		soma += d * d
	}
	return math.Sqrt(soma)
}

// =============================================================================
// Forward pass (propagação direta)
//
// Do slide: propagar o sinal camada por camada com ativação tanh.
//
// Camada oculta:
//   z_in_j = V0[j] + Σ_i (x[i] * V[i][j])    — combinação linear
//   z[j]   = tanh(z_in_j)                       — ativação
//
// Camada de saída (tanh puro, SEM limiar — diferente do MLP Letras):
//   y_in_k = W0[k] + Σ_j (z[j] * W[j][k])
//   y[k]   = tanh(y_in_k)
//
// A saída y é um vetor contínuo de 32 valores em [-1, +1].
// =============================================================================

func forward(m OrtMLP, x [NIn]float64) (z []float64, y []float64) {
	z = make([]float64, m.nHid)
	y = make([]float64, NOrt)

	// Camada oculta: z = tanh(V0 + x·V)
	for j := range m.nHid {
		zin := m.V0[j] // bias
		for i := range NIn {
			zin += x[i] * m.V[i][j] // soma ponderada
		}
		z[j] = math.Tanh(zin) // ativação tanh
	}

	// Camada de saída: y = tanh(W0 + z·W) — sem limiar!
	for k := range NOrt {
		yin := m.W0[k] // bias
		for j := range m.nHid {
			yin += z[j] * m.W[j][k] // soma ponderada
		}
		y[k] = math.Tanh(yin) // ativação tanh
	}
	return
}

// =============================================================================
// Backward pass + atualização de pesos (backpropagation)
//
// Regra Delta Generalizada (do slide):
//
// 1. Delta da camada de saída:
//    δ_k = (t_k - y_k) * f'(y_k)     onde f'(y) = (1+y)(1-y) para tanh
//
// 2. Propagar delta para camada oculta:
//    δ_j = (Σ_k δ_k * W[j][k]) * f'(z_j)
//
// 3. Atualizar pesos (regra delta):
//    W[j][k] += α * δ_k * z[j]     (pesos oculta→saída)
//    W0[k]   += α * δ_k             (bias saída)
//    V[i][j] += α * δ_j * x[i]     (pesos entrada→oculta)
//    V0[j]   += α * δ_j             (bias oculta)
// =============================================================================

// tanhDeriv calcula a derivada da tanh: f'(y) = (1+y)(1-y)
func tanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func backwardAndUpdate(m *OrtMLP, z []float64, y []float64, target [NOrt]float64, x [NIn]float64, alfa float64) {
	// 1. Delta da camada de saída: δ_k = (target_k - y_k) * f'(y_k)
	deltaK := make([]float64, NOrt)
	for k := range NOrt {
		deltaK[k] = (target[k] - y[k]) * tanhDeriv(y[k])
	}

	// 2. Propagar delta para camada oculta: δ_j = (Σ δ_k * W[j][k]) * f'(z_j)
	deltaJ := make([]float64, m.nHid)
	for j := range m.nHid {
		var s float64
		for k := range NOrt {
			s += deltaK[k] * m.W[j][k] // erro propagado da saída
		}
		deltaJ[j] = s * tanhDeriv(z[j])
	}

	// 3. Atualizar pesos W (oculta → saída): ΔW = α * δ_k * z_j
	for j := range m.nHid {
		for k := range NOrt {
			m.W[j][k] += alfa * deltaK[k] * z[j]
		}
	}
	for k := range NOrt {
		m.W0[k] += alfa * deltaK[k] // bias saída
	}

	// 4. Atualizar pesos V (entrada → oculta): ΔV = α * δ_j * x_i
	for i := range NIn {
		for j := range m.nHid {
			m.V[i][j] += alfa * deltaJ[j] * x[i]
		}
	}
	for j := range m.nHid {
		m.V0[j] += alfa * deltaJ[j] // bias oculta
	}
}

// =============================================================================
// Erro quadrático
//
// E = 0.5 * Σ_k (t_k - y_k)²
//
// Usado como critério de convergência (erro total sobre todas as letras).
// =============================================================================

func calcErro(y []float64, t [NOrt]float64) float64 {
	var e float64
	for k := range NOrt {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

// =============================================================================
// Classificação por distância euclidiana
//
// Do slide: "Ao reconhecer um padrão, usamos a MENOR distância euclidiana"
//
// Calcula D(y, t_i) para cada letra i = A..Z.
// A letra com menor distância é a classificação.
// =============================================================================

func classificar(y []float64, vetores [NOrt][NOrt]float64) (int, [NClasses]float64) {
	var distancias [NClasses]float64
	best := 0
	bestDist := math.MaxFloat64
	for i := range NClasses {
		distancias[i] = distanciaEuclidiana(y, vetores[i])
		if distancias[i] < bestDist {
			bestDist = distancias[i]
			best = i
		}
	}
	return best, distancias
}

// =============================================================================
// Treinamento principal
//
// Algoritmo (do slide Aula 06):
//   Para cada ciclo (época):
//     Para cada letra (A..Z):
//       1. x = grade 5×7 da letra (35 entradas bipolares)
//       2. target = vetor ortogonal de 32 dims atribuído à letra
//       3. Forward: z, y = forward(x)
//       4. Erro: E += 0.5 * Σ(t_k - y_k)²
//       5. Backward: atualizar V, W com backpropagation
//     Se erro total <= 0.5 → CONVERGIU
//
// Após treinamento: testar acurácia com distância euclidiana.
// =============================================================================

func Treinar(progressCh chan<- OrtStep, cfg Config) (OrtResult, OrtMLP) {
	if cfg.NHid <= 0 {
		cfg = DefaultConfig()
	}

	dataset := Dataset()                // 26 letras, cada uma = [35]float64
	vetores := GerarVetoresOrtogonais() // 32 vetores ortogonais de 32 dims
	m := Inicializar(cfg.NHid)          // pesos aleatórios [-0.5, +0.5]

	var res OrtResult
	res.Vetores = vetores
	const maxSteps = 200 // limitar SSE para não sobrecarregar

	var steps []OrtStep
	erroAlvo := 0.5 // critério de convergência

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		erroTotal := 0.0

		// Apresentar todas as 26 letras à rede
		for letraIdx := range NClasses {
			x := dataset[letraIdx]      // entrada: grade 5×7 bipolar
			target := vetores[letraIdx]  // target: vetor ortogonal da letra

			z, y := forward(m, x)                              // forward pass
			erroTotal += calcErro(y, target)                    // acumular erro
			backwardAndUpdate(&m, z, y, target, x, cfg.Alfa)   // backpropagation

			// Enviar progresso via SSE (limitado a maxSteps mensagens)
			if len(steps) < maxSteps {
				step := OrtStep{
					Ciclo:       ciclo,
					LetraIdx:    letraIdx,
					Letra:       Nomes[letraIdx],
					ErroTotal:   erroTotal,
					ActiveLayer: ciclo % 2, // alternar animação entre camadas V e W
				}
				steps = append(steps, step)
				if progressCh != nil {
					select {
					case progressCh <- step:
					default:
					}
				}
			}
		}
		res.ErroHistorico = append(res.ErroHistorico, erroTotal)

		// Critério de parada
		if erroTotal <= erroAlvo {
			res.Convergiu = true
			res.Ciclos = ciclo
			res.ErroFinal = erroTotal
			break
		}
	}

	if !res.Convergiu {
		res.Ciclos = cfg.MaxCiclo
		res.ErroFinal = res.ErroHistorico[len(res.ErroHistorico)-1]
	}

	// Testar acurácia: classificar cada letra usando distância euclidiana
	for i := range NClasses {
		_, y := forward(m, dataset[i])
		best, _ := classificar(y, vetores)
		if best == i {
			res.Acertos++ // a rede acertou a letra
		}
	}
	res.Total = NClasses
	res.Acuracia = float64(res.Acertos) / float64(NClasses) * 100.0

	return res, m
}

// Classificar classifica uma entrada (grade 5×7) usando a rede treinada.
// Retorna a letra mais provável + distâncias para todos os vetores-alvo + top-5.
func Classificar(m OrtMLP, x [NIn]float64) ClassifyResp {
	vetores := GerarVetoresOrtogonais()
	_, y := forward(m, x)
	best, distancias := classificar(y, vetores)

	// Encontrar top-5 por menor distância (selection sort parcial)
	type distIdx struct {
		dist float64
		idx  int
	}
	items := make([]distIdx, NClasses)
	for i := range NClasses {
		items[i] = distIdx{distancias[i], i}
	}
	var top5 []OrtCandidate
	for i := range 5 {
		minJ := i
		for j := i + 1; j < NClasses; j++ {
			if items[j].dist < items[minJ].dist {
				minJ = j
			}
		}
		items[i], items[minJ] = items[minJ], items[i]
		top5 = append(top5, OrtCandidate{
			Letra:     Nomes[items[i].idx],
			Distancia: items[i].dist,
			Idx:       items[i].idx,
		})
	}

	return ClassifyResp{
		LetraIdx:   best,
		Letra:      Nomes[best],
		Distancias: distancias,
		Top5:       top5,
		SaidaRede:  y,
	}
}

// =============================================================================
// Info do dataset para o frontend
// =============================================================================

// DatasetInfo retorna o dataset formatado + vetores ortogonais para o frontend
type DatasetInfo struct {
	Letras  []LetraInfo         `json:"letras"`
	Vetores [NOrt][NOrt]float64 `json:"vetores"`
}

type LetraInfo struct {
	Nome  string        `json:"nome"`
	Grade [NIn]float64  `json:"grade"`
	Vetor [NOrt]float64 `json:"vetor"`
}

func GetDatasetInfo() DatasetInfo {
	dataset := Dataset()
	vetores := GerarVetoresOrtogonais()
	var info DatasetInfo
	info.Vetores = vetores
	for i := range NClasses {
		info.Letras = append(info.Letras, LetraInfo{
			Nome:  Nomes[i],
			Grade: dataset[i],
			Vetor: vetores[i],
		})
	}
	return info
}
