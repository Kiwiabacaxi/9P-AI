package genetico

import (
	"math"
	"math/rand"
)

// =============================================================================
// Algoritmo Genético — Aula 10
//
// Otimiza a função:
//
//     f(x) = -|x · sin(√|x|)|     no intervalo [0, 512]
//
// Como x ∈ [0, 512] (sempre ≥ 0), |x| = x e √|x| = √x. O objetivo é
// MINIMIZAR f, ou seja, encontrar x* que dê o valor mais negativo (variante
// da função de Schwefel — mínimo global perto de x ≈ 420.97 com f(x*) ≈ -418.98).
//
// Fluxo (fiel ao pseudocódigo do slide):
//
//     popBin       = gerarElementosBinarios(popDec)
//     imagemFuncao = gerarImagem(popDec)
//     probabRolet  = gerarProbabilidades(imagemFuncao)
//     sMelhores    = separarVinteMelhores(probabRolet, popBin)
//     casaisFormados = sortearCasais()
//     pontoCorte   = gerarPontoDeCorte()
//     s_filhos     = cruzamento(pontoCorte, casaisFormados, sMelhores)
//     s_filhos     = efetuarMutacao(s_filhos)
//
// Decisões sobre pontos ambíguos do slide:
//   - "separarVinteMelhores" recebe `probabRolet`, então interpretamos como
//     N sorteios via roleta (a roleta já privilegia indivíduos de melhor fitness).
//   - Não há elitismo: filhos substituem a população inteira (fiel ao slide).
//   - Mutação só nos filhos (Obs 03).
//   - Um único `pontoCorte` é sorteado por geração e usado em todos os casais
//     (lendo o pseudocódigo literalmente). É raro na literatura — em práticas
//     comuns sorteia-se um por casal — mas mantemos a fidelidade educativa.
// =============================================================================

const (
	dominioMin = 0.0
	dominioMax = 512.0
)

// Config — hiperparâmetros configuráveis pelo frontend.
type Config struct {
	Bits           int     `json:"bits"`           // bits por cromossomo (recomendado: 10)
	PopSize        int     `json:"popSize"`        // tamanho da população (slide usa 20)
	MaxGeracoes    int     `json:"maxGeracoes"`    // número de iterações
	ProbCruzamento float64 `json:"probCruzamento"` // 0..1 — Obs 02
	ProbMutacao    float64 `json:"probMutacao"`    // 0..1 (por bit) — Obs 01
	Seed           int64   `json:"seed,omitempty"` // 0 = aleatório
}

// DefaultConfig — valores padrão (alinhados com o slide).
func DefaultConfig() Config {
	return Config{
		Bits:           10,
		PopSize:        20,
		MaxGeracoes:    100,
		ProbCruzamento: 0.8,
		ProbMutacao:    0.05,
	}
}

// Individuo guarda o cromossomo binário, sua representação decimal e o
// valor de x já mapeado para o intervalo [0, 512].
//
// Ex: bits = [1,0,0,1,1,0,1,1,0,0]  →  dec = 604  →  x = 604/1023 * 512 ≈ 302.36
type Individuo struct {
	Bits    []int   `json:"bits"`
	Dec     int     `json:"dec"`
	X       float64 `json:"x"`
	Fitness float64 `json:"fitness"` // fitness positivo: -f(x) = |x sin(√x)|
	Fx      float64 `json:"fx"`      // valor de f(x) (sempre ≤ 0 neste domínio)
}

// Step — payload enviado por SSE a cada geração (alimenta a animação no frontend).
type Step struct {
	Geracao     int         `json:"geracao"`
	MelhorX     float64     `json:"melhorX"`
	MelhorFx    float64     `json:"melhorFx"`
	MediaFx     float64     `json:"mediaFx"`
	PiorFx      float64     `json:"piorFx"`
	Populacao   []Individuo `json:"populacao"`   // todos os indivíduos da geração
	MelhorIndiv Individuo   `json:"melhorIndiv"` // o melhor desta geração
}

// Result — resultado final do treino, retornado quando o stream termina.
type Result struct {
	Geracoes       int       `json:"geracoes"`
	MelhorX        float64   `json:"melhorX"`
	MelhorFx       float64   `json:"melhorFx"`
	MelhorIndiv    Individuo `json:"melhorIndiv"`
	HistMelhorFx   []float64 `json:"histMelhorFx"`
	HistMediaFx    []float64 `json:"histMediaFx"`
	Bits           int       `json:"bits"`
	PopSize        int       `json:"popSize"`
	ProbCruzamento float64   `json:"probCruzamento"`
	ProbMutacao    float64   `json:"probMutacao"`
}

// =============================================================================
// f(x) = -|x · sin(√|x|)|
// =============================================================================

// F é a função objetivo do slide. Use F(x) para inspeção/teste.
func F(x float64) float64 {
	return -math.Abs(x * math.Sin(math.Sqrt(math.Abs(x))))
}

// =============================================================================
// Conversão binário ↔ decimal e mapeamento para [0, 512].
//
// "gerarElementosBinarios" no slide converte uma representação para outra.
// Aqui mantemos os Bits como representação autoritativa e recalculamos
// Dec/X a partir deles sempre que a população muda (após cruzamento/mutação).
// =============================================================================

func bitsParaDecimal(bits []int) int {
	dec := 0
	for _, b := range bits {
		dec = (dec << 1) | (b & 1)
	}
	return dec
}

func decimalParaX(dec, bits int) float64 {
	maxDec := (1 << bits) - 1 // 2^bits - 1
	if maxDec == 0 {
		return dominioMin
	}
	return float64(dec)/float64(maxDec)*(dominioMax-dominioMin) + dominioMin
}

// gerarElementosBinarios — atualiza Dec/X de cada indivíduo a partir dos Bits.
// (No slide, sincroniza popDec ↔ popBin; aqui sincroniza Bits → Dec/X.)
func gerarElementosBinarios(pop []Individuo, bits int) {
	for i := range pop {
		pop[i].Dec = bitsParaDecimal(pop[i].Bits)
		pop[i].X = decimalParaX(pop[i].Dec, bits)
	}
}

// =============================================================================
// Inicialização: bits aleatórios.
// =============================================================================

func gerarPopulacaoInicial(rng *rand.Rand, popSize, bits int) []Individuo {
	pop := make([]Individuo, popSize)
	for i := range pop {
		pop[i].Bits = make([]int, bits)
		for j := 0; j < bits; j++ {
			pop[i].Bits[j] = rng.Intn(2)
		}
	}
	gerarElementosBinarios(pop, bits)
	return pop
}

// =============================================================================
// gerarImagem — calcula f(x) e o fitness de cada indivíduo.
//
// Como queremos MINIMIZAR f e f(x) ≤ 0 neste domínio, usamos
// fitness = -f(x) ≥ 0, de modo que indivíduos com f mais negativo
// (melhores) recebam fitness MAIOR — exatamente o que a roleta privilegia.
// =============================================================================

func gerarImagem(pop []Individuo) {
	for i := range pop {
		pop[i].Fx = F(pop[i].X)
		pop[i].Fitness = -pop[i].Fx
	}
}

// =============================================================================
// gerarProbabilidades — fitness → probabilidade de roleta.
//
//     prob_i = fitness_i / Σ fitness
//
// Caso degenerado (soma zero): distribui uniformemente.
// =============================================================================

func gerarProbabilidades(pop []Individuo) []float64 {
	probs := make([]float64, len(pop))
	soma := 0.0
	for _, ind := range pop {
		soma += ind.Fitness
	}
	if soma == 0 {
		for i := range probs {
			probs[i] = 1.0 / float64(len(pop))
		}
		return probs
	}
	for i, ind := range pop {
		probs[i] = ind.Fitness / soma
	}
	return probs
}

// =============================================================================
// separarVinteMelhores — sorteia `n` indivíduos via roleta (com reposição).
//
// Apesar do nome do slide, a função recebe `probabRolet` e usa seleção
// proporcional ao fitness — a roleta privilegia naturalmente os melhores.
// Pode haver indivíduo selecionado mais de uma vez (típico em roleta).
//
// Algoritmo:
//   1. Calcula a soma cumulativa das probabilidades.
//   2. Sorteia um número r ∈ [0, somaTotal).
//   3. Encontra o primeiro índice cuja cumul[i] >= r — esse é o selecionado.
//   4. Repete n vezes.
// =============================================================================

func separarVinteMelhores(probs []float64, pop []Individuo, n int, rng *rand.Rand) []Individuo {
	cumul := make([]float64, len(probs))
	soma := 0.0
	for i, p := range probs {
		soma += p
		cumul[i] = soma
	}
	sel := make([]Individuo, n)
	for k := 0; k < n; k++ {
		r := rng.Float64() * cumul[len(cumul)-1]
		idx := 0
		for idx < len(cumul)-1 && cumul[idx] < r {
			idx++
		}
		// cópia profunda (vamos modificar nos próximos passos)
		sel[k] = clonarIndividuo(pop[idx])
	}
	return sel
}

func clonarIndividuo(src Individuo) Individuo {
	bits := make([]int, len(src.Bits))
	copy(bits, src.Bits)
	return Individuo{
		Bits:    bits,
		Dec:     src.Dec,
		X:       src.X,
		Fitness: src.Fitness,
		Fx:      src.Fx,
	}
}

// =============================================================================
// sortearCasais — embaralha a população selecionada e empareia 0-1, 2-3, ...
// =============================================================================

// Casal — par de índices na lista de selecionados.
type Casal struct {
	A int
	B int
}

func sortearCasais(n int, rng *rand.Rand) []Casal {
	idx := rng.Perm(n)
	casais := make([]Casal, 0, n/2)
	for i := 0; i+1 < n; i += 2 {
		casais = append(casais, Casal{A: idx[i], B: idx[i+1]})
	}
	return casais
}

// =============================================================================
// gerarPontoDeCorte — int aleatório em [1, bits-1].
// (Não pode ser 0 nem bits — senão o "corte" não troca nada.)
// =============================================================================

func gerarPontoDeCorte(bits int, rng *rand.Rand) int {
	if bits < 2 {
		return 1
	}
	return 1 + rng.Intn(bits-1)
}

// =============================================================================
// cruzamento — para cada casal, com prob `probCruzamento`, troca os bits
// DEPOIS do `pontoCorte` entre os dois pais. Caso contrário, filhos são
// cópias dos pais. (Obs 02: probCruzamento = porcentagem de elementos que
// efetivamente cruzam.)
//
// Exemplo (bits=10, pontoCorte=4):
//
//     Pai A: 1010 | 110100
//     Pai B: 0011 | 001011
//                ^
//     Filho A: 1010 | 001011
//     Filho B: 0011 | 110100
// =============================================================================

func cruzamento(pontoCorte int, casais []Casal, pais []Individuo, probCruzamento float64, rng *rand.Rand) []Individuo {
	filhos := make([]Individuo, 0, len(pais))
	for _, c := range casais {
		paiA := pais[c.A]
		paiB := pais[c.B]
		filhoA := clonarIndividuo(paiA)
		filhoB := clonarIndividuo(paiB)
		if rng.Float64() < probCruzamento {
			for k := pontoCorte; k < len(paiA.Bits); k++ {
				filhoA.Bits[k] = paiB.Bits[k]
				filhoB.Bits[k] = paiA.Bits[k]
			}
		}
		filhos = append(filhos, filhoA, filhoB)
	}
	return filhos
}

// =============================================================================
// efetuarMutacao — para cada bit dos filhos, com prob `probMutacao`, flipa.
// (Obs 03: mutação só nos filhos — pais selecionados não são modificados.)
// =============================================================================

func efetuarMutacao(filhos []Individuo, probMutacao float64, rng *rand.Rand) {
	for i := range filhos {
		for j := range filhos[i].Bits {
			if rng.Float64() < probMutacao {
				filhos[i].Bits[j] = 1 - filhos[i].Bits[j]
			}
		}
	}
}

// =============================================================================
// Treinar — orquestra o algoritmo. Emite um Step por geração via canal.
// O canal NÃO é fechado por esta função (responsabilidade do chamador, que
// pode querer sinalizar "fim" via mecanismo separado, como SSE event=done).
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config) Result {
	// sanitização defensiva
	if cfg.Bits <= 0 {
		cfg.Bits = 10
	}
	if cfg.PopSize <= 1 {
		cfg.PopSize = 20
	}
	if cfg.PopSize%2 != 0 {
		cfg.PopSize++ // garante pares (necessário para casais)
	}
	if cfg.MaxGeracoes <= 0 {
		cfg.MaxGeracoes = 100
	}
	if cfg.ProbCruzamento < 0 {
		cfg.ProbCruzamento = 0
	}
	if cfg.ProbCruzamento > 1 {
		cfg.ProbCruzamento = 1
	}
	if cfg.ProbMutacao < 0 {
		cfg.ProbMutacao = 0
	}
	if cfg.ProbMutacao > 1 {
		cfg.ProbMutacao = 1
	}
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	// população inicial aleatória
	pop := gerarPopulacaoInicial(rng, cfg.PopSize, cfg.Bits)

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)

	// rastreia o melhor global ao longo de TODAS as gerações
	// (sem isso, sem elitismo, podemos "perder" o melhor de uma geração pra outra)
	var melhorGlobal Individuo
	melhorGlobal.Fitness = math.Inf(-1)

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// 1) imagem da função: calcula f(x) e fitness pra cada indivíduo
		gerarImagem(pop)

		// 2) probabilidades da roleta
		probs := gerarProbabilidades(pop)

		// estatísticas da geração atual
		melhorIdx := 0
		soma := 0.0
		piorFx := math.Inf(-1)
		for i, ind := range pop {
			soma += ind.Fx
			if ind.Fitness > pop[melhorIdx].Fitness {
				melhorIdx = i
			}
			if ind.Fx > piorFx {
				piorFx = ind.Fx // mais "alto" = pior, já que minimizamos
			}
		}
		mediaFx := soma / float64(len(pop))
		melhorFx := pop[melhorIdx].Fx
		histMelhor = append(histMelhor, melhorFx)
		histMedia = append(histMedia, mediaFx)
		if pop[melhorIdx].Fitness > melhorGlobal.Fitness {
			melhorGlobal = clonarIndividuo(pop[melhorIdx])
		}

		// emite step pro frontend (animação)
		if progressCh != nil {
			progressCh <- Step{
				Geracao:     g + 1,
				MelhorX:     pop[melhorIdx].X,
				MelhorFx:    melhorFx,
				MediaFx:     mediaFx,
				PiorFx:      piorFx,
				Populacao:   clonarPop(pop),
				MelhorIndiv: clonarIndividuo(pop[melhorIdx]),
			}
		}

		// 3) seleção via roleta — sorteia popSize indivíduos (com reposição)
		sMelhores := separarVinteMelhores(probs, pop, cfg.PopSize, rng)

		// 4) emparelhar em casais (popSize/2 casais)
		casais := sortearCasais(cfg.PopSize, rng)

		// 5) cruzamento — pontoCorte sorteado uma vez por geração (fiel ao slide)
		pontoCorte := gerarPontoDeCorte(cfg.Bits, rng)
		filhos := cruzamento(pontoCorte, casais, sMelhores, cfg.ProbCruzamento, rng)

		// 6) mutação (só nos filhos — Obs 03)
		efetuarMutacao(filhos, cfg.ProbMutacao, rng)

		// próxima geração: filhos substituem a população inteira (sem elitismo)
		pop = filhos
		gerarElementosBinarios(pop, cfg.Bits)
	}

	return Result{
		Geracoes:       cfg.MaxGeracoes,
		MelhorX:        melhorGlobal.X,
		MelhorFx:       melhorGlobal.Fx,
		MelhorIndiv:    melhorGlobal,
		HistMelhorFx:   histMelhor,
		HistMediaFx:    histMedia,
		Bits:           cfg.Bits,
		PopSize:        cfg.PopSize,
		ProbCruzamento: cfg.ProbCruzamento,
		ProbMutacao:    cfg.ProbMutacao,
	}
}

func clonarPop(pop []Individuo) []Individuo {
	out := make([]Individuo, len(pop))
	for i := range pop {
		out[i] = clonarIndividuo(pop[i])
	}
	return out
}
