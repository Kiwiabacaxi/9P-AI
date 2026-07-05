package fuzzy

import (
	"fmt"
	"math"
)

// =============================================================================
// Lógica Fuzzy — Qualidade da Água (potabilidade)
// Trabalho 16 — Aulas 17–19 · exemplo 2.7.3 da apostila de Fuzzy
// (Jafelice · Barros · Bassanezi), com limites da SABESP.
//
// Sistema de inferência MAMDANI clássico:
//
//   1. FUZZIFICAÇÃO — cada entrada nítida (cor aparente, pH, turbidez) vira um
//      grau de pertinência μ ∈ [0,1] em cada termo linguístico (trapézios).
//   2. REGRAS — 45 regras (Tabelas 2.6/2.7/2.8 da apostila) no formato
//      SE aparência é X E pH é Y E turbidez é Z ENTÃO qualidade é W.
//      O conectivo E é o MÍNIMO: força da regra = min(μ_aparência, μ_pH, μ_turb).
//   3. IMPLICAÇÃO + AGREGAÇÃO — cada termo de saída é RECORTADO (min) na força
//      máxima das regras que apontam pra ele; o envelope final é o MÁXIMO.
//   4. DEFUZZIFICAÇÃO — centroide da área agregada, discretizado em [0,1].
//
// Os cantos dos trapézios foram fixados pra que os cruzamentos em μ = 0.5 caiam
// EXATAMENTE nos limites da SABESP (cor 5 e 15 UH · pH 6, 6.5, 8.5, 10 ·
// turbidez 1 e 5 UT). Com isso o exemplo canônico da apostila fecha redondo:
// cor 15, pH 7, turbidez 0 → duas regras disparam com força 0.5, ambas em
// "adequada", e o trapézio recortado é simétrico em torno de 0.6 → Q = 0.6.
// =============================================================================

// Trapezio — função de pertinência trapezoidal (a, b, c, d):
// μ = 0 até a, sobe linear até b, platô 1 até c, desce linear até d.
// Ombros nas bordas do domínio usam a == b (esquerdo) ou c == d (direito).
type Trapezio [4]float64

// Mu — grau de pertinência de x no trapézio.
func (t Trapezio) Mu(x float64) float64 {
	a, b, c, d := t[0], t[1], t[2], t[3]
	switch {
	case x < a || x > d:
		return 0
	case x >= b && x <= c:
		return 1
	case x < b: // rampa de subida (aqui b > a, senão cairia no caso anterior)
		return (x - a) / (b - a)
	default: // rampa de descida (d > c garantido)
		return (d - x) / (d - c)
	}
}

// Termo — um termo linguístico de uma variável ("boa", "bom", "inadequada"…).
type Termo struct {
	ID   string   `json:"id"`
	Nome string   `json:"nome"`
	Trap Trapezio `json:"trap"`
	Cor  string   `json:"cor"` // cor sugerida pra UI
}

// Variavel — variável linguística com domínio e termos.
type Variavel struct {
	ID      string  `json:"id"`
	Nome    string  `json:"nome"`
	Unidade string  `json:"unidade"`
	Min     float64 `json:"min"`
	Max     float64 `json:"max"`
	Termos  []Termo `json:"termos"`
}

// Fuzzificar — μ de x em cada termo da variável (x é clampado ao domínio).
func (v Variavel) Fuzzificar(x float64) map[string]float64 {
	x = clamp(x, v.Min, v.Max)
	out := make(map[string]float64, len(v.Termos))
	for _, t := range v.Termos {
		out[t.ID] = t.Trap.Mu(x)
	}
	return out
}

// =============================================================================
// As variáveis do problema (limites SABESP / Figuras 2.10–2.13 da apostila)
// =============================================================================

const (
	corBoa        = "boa"
	corAdequada   = "adequada"
	corInadequada = "inadequada"

	phInadBaixo = "inadequadoBaixo"
	phAdeqBaixo = "adequadoBaixo"
	phBom       = "bom"
	phAdeqAlto  = "adequadoAlto"
	phInadAlto  = "inadequadoAlto"

	// saída
	QInadequada = "inadequada"
	QAdequada   = "adequada"
	QBoa        = "boa"
)

// Cores semânticas da UI: verde = bom, âmbar = adequado, vermelho = inadequado.
const (
	corVerde    = "#3ddc84"
	corAmbar    = "#ffb020"
	corVermelho = "#ff4d6d"
)

var VarCor = Variavel{
	ID: "cor", Nome: "Cor aparente", Unidade: "UH", Min: 0, Max: 30,
	Termos: []Termo{
		{ID: corBoa, Nome: "boa", Trap: Trapezio{0, 0, 4, 6}, Cor: corVerde},
		{ID: corAdequada, Nome: "adequada", Trap: Trapezio{4, 6, 14, 16}, Cor: corAmbar},
		{ID: corInadequada, Nome: "inadequada", Trap: Trapezio{14, 16, 30, 30}, Cor: corVermelho},
	},
}

var VarPH = Variavel{
	ID: "ph", Nome: "pH", Unidade: "", Min: 0, Max: 14,
	Termos: []Termo{
		{ID: phInadBaixo, Nome: "inadequado baixo", Trap: Trapezio{0, 0, 5.75, 6.25}, Cor: corVermelho},
		{ID: phAdeqBaixo, Nome: "adequado baixo", Trap: Trapezio{5.75, 6.25, 6.25, 6.75}, Cor: corAmbar},
		{ID: phBom, Nome: "bom", Trap: Trapezio{6.25, 6.75, 8.25, 8.75}, Cor: corVerde},
		{ID: phAdeqAlto, Nome: "adequado alto", Trap: Trapezio{8.25, 8.75, 9.75, 10.25}, Cor: corAmbar},
		{ID: phInadAlto, Nome: "inadequado alto", Trap: Trapezio{9.75, 10.25, 14, 14}, Cor: corVermelho},
	},
}

var VarTurbidez = Variavel{
	ID: "turbidez", Nome: "Turbidez", Unidade: "UT", Min: 0, Max: 10,
	Termos: []Termo{
		{ID: corBoa, Nome: "boa", Trap: Trapezio{0, 0, 0.8, 1.2}, Cor: corVerde},
		{ID: corAdequada, Nome: "adequada", Trap: Trapezio{0.8, 1.2, 4.6, 5.4}, Cor: corAmbar},
		{ID: corInadequada, Nome: "inadequada", Trap: Trapezio{4.6, 5.4, 10, 10}, Cor: corVermelho},
	},
}

var VarQualidade = Variavel{
	ID: "qualidade", Nome: "Qualidade da água", Unidade: "", Min: 0, Max: 1,
	Termos: []Termo{
		{ID: QInadequada, Nome: "inadequada", Trap: Trapezio{0, 0, 0.35, 0.5}, Cor: corVermelho},
		{ID: QAdequada, Nome: "adequada", Trap: Trapezio{0.35, 0.5, 0.7, 0.85}, Cor: corAmbar},
		{ID: QBoa, Nome: "boa", Trap: Trapezio{0.7, 0.85, 1, 1}, Cor: corVerde},
	},
}

// =============================================================================
// Base de regras — Tabelas 2.6, 2.7 e 2.8 da apostila, declaradas como DADOS.
// Linhas = termo do pH · colunas = termo da turbidez (boa, adequada, inadequada).
// =============================================================================

var ordemPH = []string{phInadBaixo, phAdeqBaixo, phBom, phAdeqAlto, phInadAlto}
var ordemTurb = []string{corBoa, corAdequada, corInadequada}

// tabelas[aparência][pH][turbidez] → termo de saída.
var tabelas = map[string][5][3]string{
	// Tabela 2.6 — aparência da água é BOA
	corBoa: {
		{QInadequada, QInadequada, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QBoa, QBoa, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QInadequada, QInadequada, QInadequada},
	},
	// Tabela 2.7 — aparência da água é ADEQUADA
	corAdequada: {
		{QInadequada, QInadequada, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QInadequada, QInadequada, QInadequada},
	},
	// Tabela 2.8 — aparência da água é INADEQUADA
	corInadequada: {
		{QInadequada, QInadequada, QInadequada},
		{QInadequada, QInadequada, QInadequada},
		{QAdequada, QAdequada, QInadequada},
		{QInadequada, QInadequada, QInadequada},
		{QInadequada, QInadequada, QInadequada},
	},
}

// Regra — SE aparência é X E pH é Y E turbidez é Z ENTÃO qualidade é W.
type Regra struct {
	Aparencia string `json:"aparencia"`
	PH        string `json:"ph"`
	Turbidez  string `json:"turbidez"`
	Saida     string `json:"saida"`
}

// Regras — as 45 regras achatadas na ordem (aparência, pH, turbidez).
var Regras = montarRegras()

func montarRegras() []Regra {
	out := make([]Regra, 0, 45)
	for _, ap := range []string{corBoa, corAdequada, corInadequada} {
		tab := tabelas[ap]
		for i, ph := range ordemPH {
			for j, tb := range ordemTurb {
				out = append(out, Regra{Aparencia: ap, PH: ph, Turbidez: tb, Saida: tab[i][j]})
			}
		}
	}
	return out
}

// =============================================================================
// Inferência
// =============================================================================

// Entrada — valores nítidos medidos.
type Entrada struct {
	Cor      float64 `json:"cor"`
	PH       float64 `json:"ph"`
	Turbidez float64 `json:"turbidez"`
}

// RegraAtivada — uma regra com o trace da sua ativação.
type RegraAtivada struct {
	Regra
	MuAparencia float64 `json:"muAparencia"`
	MuPH        float64 `json:"muPh"`
	MuTurbidez  float64 `json:"muTurbidez"`
	Forca       float64 `json:"forca"` // min dos três antecedentes
}

// PontoCurva — amostra da saída pra desenhar: cada termo RECORTADO + envelope.
type PontoCurva struct {
	X          float64 `json:"x"`
	Inadequada float64 `json:"inadequada"`
	Adequada   float64 `json:"adequada"`
	Boa        float64 `json:"boa"`
	Agregada   float64 `json:"agregada"`
}

// Resultado — trace completo de uma inferência (a view desenha tudo disto).
type Resultado struct {
	Entrada      Entrada                       `json:"entrada"`
	Pertinencias map[string]map[string]float64 `json:"pertinencias"` // varID → termoID → μ
	Regras       []RegraAtivada                `json:"regras"`
	ForcaSaida   map[string]float64            `json:"forcaSaida"` // termoID → força máx (agregação)
	Curva        []PontoCurva                  `json:"curva"`
	Centroide    float64                       `json:"centroide"`
	Classe       string                        `json:"classe"`
	RegrasAtivas int                           `json:"regrasAtivas"`
}

const (
	nCentroide = 1001 // resolução da defuzzificação
	nCurva     = 201  // resolução da curva enviada pra UI
)

// inferir — passos 1–3 do Mamdani (fuzzificação, regras, agregação).
func inferir(e Entrada) (pert map[string]map[string]float64, ativadas []RegraAtivada, forcaSaida map[string]float64) {
	pert = map[string]map[string]float64{
		"cor":      VarCor.Fuzzificar(e.Cor),
		"ph":       VarPH.Fuzzificar(e.PH),
		"turbidez": VarTurbidez.Fuzzificar(e.Turbidez),
	}

	ativadas = make([]RegraAtivada, len(Regras))
	forcaSaida = map[string]float64{QInadequada: 0, QAdequada: 0, QBoa: 0}
	for i, r := range Regras {
		ma := pert["cor"][r.Aparencia]
		mp := pert["ph"][r.PH]
		mt := pert["turbidez"][r.Turbidez]
		f := math.Min(ma, math.Min(mp, mt))
		ativadas[i] = RegraAtivada{Regra: r, MuAparencia: ma, MuPH: mp, MuTurbidez: mt, Forca: f}
		if f > forcaSaida[r.Saida] {
			forcaSaida[r.Saida] = f
		}
	}
	return pert, ativadas, forcaSaida
}

// muAgregado — envelope max dos termos de saída recortados nas suas forças.
func muAgregado(x float64, forcaSaida map[string]float64) float64 {
	mu := 0.0
	for _, t := range VarQualidade.Termos {
		m := math.Min(forcaSaida[t.ID], t.Trap.Mu(x))
		if m > mu {
			mu = m
		}
	}
	return mu
}

// centroide — defuzzificação Σx·μ / Σμ sobre [0,1] discretizado.
func centroide(forcaSaida map[string]float64) float64 {
	somaXMu, somaMu := 0.0, 0.0
	for i := 0; i < nCentroide; i++ {
		x := float64(i) / float64(nCentroide-1)
		mu := muAgregado(x, forcaSaida)
		somaXMu += x * mu
		somaMu += mu
	}
	if somaMu == 0 {
		// não acontece: os termos cobrem todo o domínio e sempre há regra ativa,
		// mas evita divisão por zero por segurança.
		return 0
	}
	return somaXMu / somaMu
}

// classificar — termo de saída com maior μ no centroide (empate: maior força).
func classificar(q float64, forcaSaida map[string]float64) string {
	melhor, melhorMu := VarQualidade.Termos[0].ID, -1.0
	for _, t := range VarQualidade.Termos {
		mu := t.Trap.Mu(q)
		if mu > melhorMu+1e-12 || (math.Abs(mu-melhorMu) <= 1e-12 && forcaSaida[t.ID] > forcaSaida[melhor]) {
			melhor, melhorMu = t.ID, mu
		}
	}
	return melhor
}

// Avaliar — pipeline Mamdani completo com trace pra visualização.
func Avaliar(e Entrada) Resultado {
	e.Cor = clamp(e.Cor, VarCor.Min, VarCor.Max)
	e.PH = clamp(e.PH, VarPH.Min, VarPH.Max)
	e.Turbidez = clamp(e.Turbidez, VarTurbidez.Min, VarTurbidez.Max)

	pert, ativadas, forcaSaida := inferir(e)

	curva := make([]PontoCurva, nCurva)
	for i := 0; i < nCurva; i++ {
		x := float64(i) / float64(nCurva-1)
		curva[i] = PontoCurva{
			X:          x,
			Inadequada: math.Min(forcaSaida[QInadequada], VarQualidade.Termos[0].Trap.Mu(x)),
			Adequada:   math.Min(forcaSaida[QAdequada], VarQualidade.Termos[1].Trap.Mu(x)),
			Boa:        math.Min(forcaSaida[QBoa], VarQualidade.Termos[2].Trap.Mu(x)),
			Agregada:   muAgregado(x, forcaSaida),
		}
	}

	q := centroide(forcaSaida)
	ativas := 0
	for _, r := range ativadas {
		if r.Forca > 0 {
			ativas++
		}
	}

	return Resultado{
		Entrada:      e,
		Pertinencias: pert,
		Regras:       ativadas,
		ForcaSaida:   forcaSaida,
		Curva:        curva,
		Centroide:    q,
		Classe:       classificar(q, forcaSaida),
		RegrasAtivas: ativas,
	}
}

// =============================================================================
// Meta (pra UI desenhar trapézios e tabelas) e superfície 3D
// =============================================================================

// Meta — definição completa do sistema; a view desenha tudo a partir disto.
type Meta struct {
	Entradas []Variavel `json:"entradas"`
	Saida    Variavel   `json:"saida"`
	Regras   []Regra    `json:"regras"`
	OrdemPH  []string   `json:"ordemPh"`   // ordem das linhas das tabelas
	OrdemTur []string   `json:"ordemTurb"` // ordem das colunas das tabelas
}

func GetMeta() Meta {
	return Meta{
		Entradas: []Variavel{VarCor, VarPH, VarTurbidez},
		Saida:    VarQualidade,
		Regras:   Regras,
		OrdemPH:  ordemPH,
		OrdemTur: ordemTurb,
	}
}

// Superficie — grade Q = f(eixoX, eixoY) com a terceira variável fixa.
type Superficie struct {
	EixoX string      `json:"eixoX"`
	EixoY string      `json:"eixoY"`
	Fixa  Entrada     `json:"fixa"` // valores usados nas variáveis não plotadas
	Xs    []float64   `json:"xs"`
	Ys    []float64   `json:"ys"`
	Z     [][]float64 `json:"z"` // Z[i][j] = Q(xs[j], ys[i]) — linha = y (padrão Plotly)
}

const nSurf = 41

var varPorID = map[string]*Variavel{"cor": &VarCor, "ph": &VarPH, "turbidez": &VarTurbidez}

// GerarSuperficie — avalia o sistema numa grade nSurf×nSurf.
func GerarSuperficie(eixoX, eixoY string, fixa Entrada) (Superficie, error) {
	vx, okX := varPorID[eixoX]
	vy, okY := varPorID[eixoY]
	if !okX || !okY {
		return Superficie{}, fmt.Errorf("eixos devem ser cor, ph ou turbidez")
	}
	if eixoX == eixoY {
		return Superficie{}, fmt.Errorf("eixoX e eixoY devem ser diferentes")
	}

	xs := linspace(vx.Min, vx.Max, nSurf)
	ys := linspace(vy.Min, vy.Max, nSurf)
	z := make([][]float64, nSurf)
	for i, y := range ys {
		z[i] = make([]float64, nSurf)
		for j, x := range xs {
			e := fixa
			setVar(&e, eixoX, x)
			setVar(&e, eixoY, y)
			_, _, forca := inferir(e)
			z[i][j] = centroide(forca)
		}
	}
	return Superficie{EixoX: eixoX, EixoY: eixoY, Fixa: fixa, Xs: xs, Ys: ys, Z: z}, nil
}

func setVar(e *Entrada, id string, v float64) {
	switch id {
	case "cor":
		e.Cor = v
	case "ph":
		e.PH = v
	case "turbidez":
		e.Turbidez = v
	}
}

func linspace(min, max float64, n int) []float64 {
	out := make([]float64, n)
	for i := range out {
		out[i] = min + (max-min)*float64(i)/float64(n-1)
	}
	return out
}

func clamp(x, min, max float64) float64 {
	if x < min {
		return min
	}
	if x > max {
		return max
	}
	return x
}
