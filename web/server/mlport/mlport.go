package mlport

import "math"

// =============================================================================
// MLP com Vetores Bipolares Ortogonais — Classificacao por Distancia Euclidiana
//
// Baseado na Aula 06: "Nao usaremos o limiar... usaremos a distancia
// Euclidiana para identificar cada elemento."
//
// Formula da distancia euclidiana (do slide):
//   D = sqrt( sum_k (t_k - y_k)^2 )
//
// "Ao reconhecer um padrao, usamos a MENOR distancia euclidiana"
// =============================================================================

// Config permite customizar hiperparametros via frontend
type Config struct {
	NHid     int     `json:"nHid"`
	Alfa     float64 `json:"alfa"`
	MaxCiclo int     `json:"maxCiclo"`
}

// DefaultConfig retorna a configuracao padrao
func DefaultConfig() Config {
	return Config{
		NHid:     15,
		Alfa:     0.01,
		MaxCiclo: 50000,
	}
}

// OrtMLP armazena os pesos da rede com tamanho dinamico.
type OrtMLP struct {
	nHid int
	V    [][]float64 // [NIn][nHid]
	V0   []float64   // [nHid]
	W    [][]float64 // [nHid][NOrt]
	W0   []float64   // [NOrt]
}

type OrtStep struct {
	Ciclo     int     `json:"ciclo"`
	LetraIdx  int     `json:"letraIdx"`
	Letra     string  `json:"letra"`
	ErroTotal float64 `json:"erroTotal"`
}

type OrtResult struct {
	Convergiu     bool              `json:"convergiu"`
	Ciclos        int               `json:"ciclos"`
	ErroFinal     float64           `json:"erroFinal"`
	ErroHistorico []float64         `json:"erroHistorico"`
	Acertos       int               `json:"acertos"`
	Total         int               `json:"total"`
	Acuracia      float64           `json:"acuracia"`
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
	LetraIdx   int              `json:"letraIdx"`
	Letra      string           `json:"letra"`
	Distancias [NClasses]float64 `json:"distancias"`
	Top5       []OrtCandidate   `json:"top5"`
	SaidaRede  []float64        `json:"saidaRede"`
}

// ----- Distancia Euclidiana -----

func distanciaEuclidiana(y []float64, t [NOrt]float64) float64 {
	var soma float64
	for k := 0; k < NOrt; k++ {
		d := t[k] - y[k]
		soma += d * d
	}
	return math.Sqrt(soma)
}

// ----- Forward pass -----

func forward(m OrtMLP, x [NIn]float64) (z []float64, y []float64) {
	z = make([]float64, m.nHid)
	y = make([]float64, NOrt)
	// Camada oculta
	for j := 0; j < m.nHid; j++ {
		zin := m.V0[j]
		for i := 0; i < NIn; i++ {
			zin += x[i] * m.V[i][j]
		}
		z[j] = math.Tanh(zin)
	}
	// Camada de saida — tanh puro, sem limiar
	for k := 0; k < NOrt; k++ {
		yin := m.W0[k]
		for j := 0; j < m.nHid; j++ {
			yin += z[j] * m.W[j][k]
		}
		y[k] = math.Tanh(yin)
	}
	return
}

// ----- Backward pass + atualizacao -----

func tanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func backwardAndUpdate(m *OrtMLP, z []float64, y []float64, target [NOrt]float64, x [NIn]float64, alfa float64) {
	// Delta camada de saida
	deltaK := make([]float64, NOrt)
	for k := 0; k < NOrt; k++ {
		deltaK[k] = (target[k] - y[k]) * tanhDeriv(y[k])
	}

	// Propaga para oculta
	deltaJ := make([]float64, m.nHid)
	for j := 0; j < m.nHid; j++ {
		var s float64
		for k := 0; k < NOrt; k++ {
			s += deltaK[k] * m.W[j][k]
		}
		deltaJ[j] = s * tanhDeriv(z[j])
	}

	// Atualiza W
	for j := 0; j < m.nHid; j++ {
		for k := 0; k < NOrt; k++ {
			m.W[j][k] += alfa * deltaK[k] * z[j]
		}
	}
	for k := 0; k < NOrt; k++ {
		m.W0[k] += alfa * deltaK[k]
	}

	// Atualiza V
	for i := 0; i < NIn; i++ {
		for j := 0; j < m.nHid; j++ {
			m.V[i][j] += alfa * deltaJ[j] * x[i]
		}
	}
	for j := 0; j < m.nHid; j++ {
		m.V0[j] += alfa * deltaJ[j]
	}
}

// ----- Erro -----

func calcErro(y []float64, t [NOrt]float64) float64 {
	var e float64
	for k := 0; k < NOrt; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

// ----- Classificacao por distancia euclidiana -----

func classificar(y []float64, vetores [NOrt][NOrt]float64) (int, [NClasses]float64) {
	var distancias [NClasses]float64
	best := 0
	bestDist := math.MaxFloat64
	for i := 0; i < NClasses; i++ {
		distancias[i] = distanciaEuclidiana(y, vetores[i])
		if distancias[i] < bestDist {
			bestDist = distancias[i]
			best = i
		}
	}
	return best, distancias
}

// ----- Treinamento -----

func Treinar(progressCh chan<- OrtStep, cfg Config) (OrtResult, OrtMLP) {
	if cfg.NHid <= 0 {
		cfg = DefaultConfig()
	}

	dataset := Dataset()
	vetores := GerarVetoresOrtogonais()
	m := Inicializar(cfg.NHid)

	var res OrtResult
	res.Vetores = vetores
	const maxSteps = 200

	var steps []OrtStep
	erroAlvo := 0.5

	for ciclo := 1; ciclo <= cfg.MaxCiclo; ciclo++ {
		erroTotal := 0.0
		for letraIdx := 0; letraIdx < NClasses; letraIdx++ {
			x := dataset[letraIdx]
			target := vetores[letraIdx]

			z, y := forward(m, x)
			erroTotal += calcErro(y, target)
			backwardAndUpdate(&m, z, y, target, x, cfg.Alfa)

			if len(steps) < maxSteps {
				step := OrtStep{
					Ciclo:     ciclo,
					LetraIdx:  letraIdx,
					Letra:     Nomes[letraIdx],
					ErroTotal: erroTotal,
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

	// Acuracia usando distancia euclidiana
	for i := 0; i < NClasses; i++ {
		_, y := forward(m, dataset[i])
		best, _ := classificar(y, vetores)
		if best == i {
			res.Acertos++
		}
	}
	res.Total = NClasses
	res.Acuracia = float64(res.Acertos) / float64(NClasses) * 100.0

	return res, m
}

// Classificar classifica uma entrada e retorna a resposta com distancias e saida bruta.
func Classificar(m OrtMLP, x [NIn]float64) ClassifyResp {
	vetores := GerarVetoresOrtogonais()
	_, y := forward(m, x)
	best, distancias := classificar(y, vetores)

	// Top-5 por menor distancia
	type distIdx struct {
		dist float64
		idx  int
	}
	items := make([]distIdx, NClasses)
	for i := 0; i < NClasses; i++ {
		items[i] = distIdx{distancias[i], i}
	}
	var top5 []OrtCandidate
	for i := 0; i < 5; i++ {
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

// DatasetInfo retorna o dataset formatado + vetores ortogonais para o frontend
type DatasetInfo struct {
	Letras  []LetraInfo           `json:"letras"`
	Vetores [NOrt][NOrt]float64   `json:"vetores"`
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
	for i := 0; i < NClasses; i++ {
		info.Letras = append(info.Letras, LetraInfo{
			Nome:  Nomes[i],
			Grade: dataset[i],
			Vetor: vetores[i],
		})
	}
	return info
}
