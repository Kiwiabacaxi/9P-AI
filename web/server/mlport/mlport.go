package mlport

import "math"

// =============================================================================
// MLP com Vetores Bipolares Ortogonais — Classificacao por Distancia Euclidiana
//
// Baseado na Aula 06: "Nao usaremos o limiar... usaremos a distancia
// Euclidiana para identificar cada elemento."
//
// Diferenca em relacao ao MLP Letras (package letras):
// - Targets: vetores bipolares ortogonais (32 dims) em vez de one-hot (26 dims)
// - Classificacao: menor distancia euclidiana em vez de argmax
// - Saida: tanh puro (sem limiar/threshold)
// - NOut: 32 (dimensao dos vetores ortogonais)
//
// Formula da distancia euclidiana (do slide):
//   D = sqrt( sum_k (t_k - y_k)^2 )
//
// "Ao reconhecer um padrao, usamos a MENOR distancia euclidiana"
// =============================================================================

const (
	NHid     = 15
	alfa     = 0.01
	maxCiclo = 50000
	erroAlvo = 0.5
)

// OrtMLP armazena os pesos da rede.
// Note que a camada de saida tem NOrt (32) neuronios, nao NClasses (26).
type OrtMLP struct {
	V  [NIn][NHid]float64  `json:"v"`
	V0 [NHid]float64       `json:"v0"`
	W  [NHid][NOrt]float64 `json:"w"`
	W0 [NOrt]float64       `json:"w0"`
}

type OrtStep struct {
	Ciclo     int     `json:"ciclo"`
	LetraIdx  int     `json:"letraIdx"`
	Letra     string  `json:"letra"`
	ErroTotal float64 `json:"erroTotal"`
}

type OrtResult struct {
	Convergiu     bool        `json:"convergiu"`
	Ciclos        int         `json:"ciclos"`
	ErroFinal     float64     `json:"erroFinal"`
	ErroHistorico []float64   `json:"erroHistorico"`
	Acertos       int         `json:"acertos"`
	Total         int         `json:"total"`
	Acuracia      float64     `json:"acuracia"`
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
}

// ----- Distancia Euclidiana -----

// distanciaEuclidiana calcula D = sqrt( sum_k (t_k - y_k)^2 )
// Esta eh a formula central da Aula 06 (slide "Um pequeno detalhe"):
// "Se a distancia euclidiana diminui, o erro tambem diminui."
func distanciaEuclidiana(y, t [NOrt]float64) float64 {
	var soma float64
	for k := 0; k < NOrt; k++ {
		d := t[k] - y[k]
		soma += d * d
	}
	return math.Sqrt(soma)
}

// ----- Forward pass -----

// forward calcula a saida da rede: tanh puro, SEM limiar.
// Slide: "Nao usaremos o limiar... sugestao: tanh(yin)"
func forward(m OrtMLP, x [NIn]float64) (z [NHid]float64, y [NOrt]float64) {
	// Camada oculta
	for j := 0; j < NHid; j++ {
		zin := m.V0[j]
		for i := 0; i < NIn; i++ {
			zin += x[i] * m.V[i][j]
		}
		z[j] = math.Tanh(zin)
	}
	// Camada de saida — tanh puro, sem limiar
	for k := 0; k < NOrt; k++ {
		yin := m.W0[k]
		for j := 0; j < NHid; j++ {
			yin += z[j] * m.W[j][k]
		}
		y[k] = math.Tanh(yin)
	}
	return
}

// ----- Backward pass + atualizacao -----

func tanhDeriv(y float64) float64 { return (1 + y) * (1 - y) }

func backwardAndUpdate(m *OrtMLP, z [NHid]float64, y [NOrt]float64, target [NOrt]float64, x [NIn]float64) {
	// Delta camada de saida
	var deltaK [NOrt]float64
	for k := 0; k < NOrt; k++ {
		deltaK[k] = (target[k] - y[k]) * tanhDeriv(y[k])
	}

	// Propaga para oculta
	var deltaJ [NHid]float64
	for j := 0; j < NHid; j++ {
		var s float64
		for k := 0; k < NOrt; k++ {
			s += deltaK[k] * m.W[j][k]
		}
		deltaJ[j] = s * tanhDeriv(z[j])
	}

	// Atualiza W
	for j := 0; j < NHid; j++ {
		for k := 0; k < NOrt; k++ {
			m.W[j][k] += alfa * deltaK[k] * z[j]
		}
	}
	for k := 0; k < NOrt; k++ {
		m.W0[k] += alfa * deltaK[k]
	}

	// Atualiza V
	for i := 0; i < NIn; i++ {
		for j := 0; j < NHid; j++ {
			m.V[i][j] += alfa * deltaJ[j] * x[i]
		}
	}
	for j := 0; j < NHid; j++ {
		m.V0[j] += alfa * deltaJ[j]
	}
}

// ----- Erro -----

func calcErro(y, t [NOrt]float64) float64 {
	var e float64
	for k := 0; k < NOrt; k++ {
		d := t[k] - y[k]
		e += d * d
	}
	return 0.5 * e
}

// ----- Classificacao por distancia euclidiana -----

// classificar encontra a letra cuja distancia euclidiana entre
// a saida da rede e o vetor ortogonal target eh a MENOR.
// Slide: "Ao reconhecer um padrao, usamos a menor distancia euclidiana"
func classificar(y [NOrt]float64, vetores [NOrt][NOrt]float64) (int, [NClasses]float64) {
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

// Treinar treina a rede com vetores ortogonais e envia progresso via SSE.
func Treinar(progressCh chan<- OrtStep) (OrtResult, OrtMLP) {
	dataset := Dataset()
	vetores := GerarVetoresOrtogonais()
	m := Inicializar()

	var res OrtResult
	res.Vetores = vetores
	const maxSteps = 200

	var steps []OrtStep

	for ciclo := 1; ciclo <= maxCiclo; ciclo++ {
		erroTotal := 0.0
		for letraIdx := 0; letraIdx < NClasses; letraIdx++ {
			x := dataset[letraIdx]
			target := vetores[letraIdx] // vetor ortogonal como target

			z, y := forward(m, x)
			erroTotal += calcErro(y, target)
			backwardAndUpdate(&m, z, y, target, x)

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
		res.Ciclos = maxCiclo
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

// Classificar classifica uma entrada e retorna a resposta com distancias.
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
	// Selection sort dos 5 menores
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
	}
}

// DatasetInfo retorna o dataset formatado + vetores ortogonais para o frontend
type DatasetInfo struct {
	Letras  []LetraInfo         `json:"letras"`
	Vetores [NOrt][NOrt]float64 `json:"vetores"`
}

type LetraInfo struct {
	Nome   string          `json:"nome"`
	Grade  [NIn]float64    `json:"grade"`
	Vetor  [NOrt]float64   `json:"vetor"`
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
