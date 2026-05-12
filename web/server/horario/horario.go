package horario

import (
	"math/rand"
	"sort"
)

// =============================================================================
// Algoritmo Genético — Horário Escolar — Aula 12
//
// Fiel ao slide: "Suponha que um AG tenha que ser modelado para elaborar o
// horário dos professores de uma escola. A representação do cromossomo pode
// ser dada por uma MATRIZ ao invés de um vetor."
//
// Cromossomo:
//   matriz[slot][turma] = id do professor
//
//   slot = dia * AulasPorDia + aulaNoDia
//   total de slots = DiasDaSemana * AulasPorDia
//
// Cruzamento (também do slide):
//   "troca de LINHAS entre indivíduos (matrizes de horários)"
//   → para cada linha (slot), filho herda a linha inteira de um dos pais.
//
// Fitness ("dependerá da criatividade"):
//   + BÔNUS por aulas encadeadas (mesma matéria do mesmo prof em slots
//     consecutivos do MESMO dia, no MÁXIMO 2 horários)
//   − PENALIDADE por choque de horário (mesmo prof em > 1 turma no mesmo slot)
//   − PENALIDADE por turmas com gargalo de professor (pouca variedade)
//
// Cada professor tem UMA matéria associada (mapeamento prof→matéria pré-fixo),
// o que permite calcular "encadeamento por matéria" mesmo guardando só o prof.
// =============================================================================

// Config — hiperparâmetros do AG + dimensões do problema.
//
// Defaults do slide (Aula 12, "Hands on!"):
//   numeroDeProfessores = 29, numeroDeTurmas = 3, aulasPorDia = 5, DiasDaSemana = 2,
//   TamanhoDaPopulacao = 100, ciclos = 10000, taxaDeMutacao = 0.1.
type Config struct {
	NumProfessores int     `json:"numProfessores"`
	NumTurmas      int     `json:"numTurmas"`
	AulasPorDia    int     `json:"aulasPorDia"`
	DiasDaSemana   int     `json:"diasDaSemana"`
	NumMaterias    int     `json:"numMaterias"` // < NumProfessores; profs são distribuídos por matéria

	PopSize        int     `json:"popSize"`
	MaxGeracoes    int     `json:"maxGeracoes"`
	ProbCruzamento float64 `json:"probCruzamento"` // por linha sorteia de qual pai
	ProbMutacao    float64 `json:"probMutacao"`    // por célula
	TamanhoTorneio int     `json:"tamanhoTorneio"`
	Elitismo       int     `json:"elitismo"`

	// Pesos da fitness (a "criatividade" do slide):
	BonusGeminada   float64 `json:"bonusGeminada"`   // + por aula encadeada válida
	PenChoque       float64 `json:"penChoque"`       // − por choque de horário
	PenSemVariedade float64 `json:"penSemVariedade"` // − por matéria que falta na turma

	Seed int64 `json:"seed,omitempty"`
}

func DefaultConfig() Config {
	return Config{
		NumProfessores:  29,
		NumTurmas:       3,
		AulasPorDia:     5,
		DiasDaSemana:    2,
		NumMaterias:     10,
		PopSize:         100,
		MaxGeracoes:     200,
		ProbCruzamento:  0.85,
		ProbMutacao:     0.10,
		TamanhoTorneio:  4,
		Elitismo:        2,
		BonusGeminada:   3.0,
		PenChoque:       10.0,
		PenSemVariedade: 1.0,
	}
}

// Professor — entidade educativa. O cromossomo só guarda o ID; o resto vive
// num catálogo lateral pra renderizar nomes/cores na UI e mapear prof→matéria
// na avaliação do fitness.
//
// Servimos sempre os DOIS nomes (código curto "P01" + nome real "Patrício")
// e deixamos o frontend escolher qual exibir via toggle.
type Professor struct {
	ID       int    `json:"id"`
	Nome     string `json:"nome"`      // código curto P01..Pnn
	NomeReal string `json:"nomeReal"`  // nome brasileiro (sorteado de catálogo)
	Materia  int    `json:"materia"`   // 0..NumMaterias-1
	MatNome  string `json:"materiaNome"`
}

// Individuo — uma matriz de horário avaliada.
//
// Matriz é flat: len = NumSlots * NumTurmas. Acesso: M[s*NumTurmas + t].
// Mantemos flat pra serializar de forma compacta em JSON.
type Individuo struct {
	Matriz   []int   `json:"matriz"`   // flat slot-major
	Fitness  float64 `json:"fitness"`
	Choques  int     `json:"choques"`
	Bonus    int     `json:"bonus"`    // # de aulas encadeadas reconhecidas
	Faltando int     `json:"faltando"` // # de (turma × matéria) sem cobertura
}

type Step struct {
	Geracao      int       `json:"geracao"`
	MelhorFit    float64   `json:"melhorFit"`
	MelhorIndiv  Individuo `json:"melhorIndiv"`
	MediaFit     float64   `json:"mediaFit"`
	Choques      int       `json:"choques"`
	Bonus        int       `json:"bonus"`
	Faltando     int       `json:"faltando"`
}

type Result struct {
	Geracoes        int         `json:"geracoes"`
	MelhorIndiv     Individuo   `json:"melhorIndiv"`
	HistMelhor      []float64   `json:"histMelhor"`
	HistMedia       []float64   `json:"histMedia"`
	HistChoques     []int       `json:"histChoques"`
	HistBonus       []int       `json:"histBonus"`
	Professores     []Professor `json:"professores"`
	Cfg             Config      `json:"cfg"`
}

// =============================================================================
// Construção do catálogo de professores
// =============================================================================

var materiasCatalogo = []string{
	"Matemática", "Português", "História", "Geografia", "Biologia",
	"Física", "Química", "Inglês", "Educação Física", "Artes",
	"Filosofia", "Sociologia", "Literatura", "Informática", "Espanhol",
}

// nomesReaisCatalogo — pool de nomes brasileiros pros professores. Quando
// numProf > len(catalogo), repete com sufixo numérico ("Patrício 2").
var nomesReaisCatalogo = []string{
	"Patrício", "Aluísio", "Heloísa", "Tarcísio", "Custódia",
	"Belmiro", "Sebastiana", "Adauto", "Iracema", "Quirino",
	"Joaquina", "Onofre", "Cândida", "Genésio", "Berenice",
	"Damião", "Eulália", "Marcílio", "Olímpia", "Severino",
	"Conceição", "Anselmo", "Ofélia", "Plácido", "Filomena",
	"Hermínio", "Gertrudes", "Raimundo", "Nazaré", "Edmundo",
	"Aparecida", "Casemiro", "Lourdes", "Inácio", "Zenaide",
	"Bartolomeu", "Etelvina", "Walfrido", "Lindalva", "Vicente",
	"Madalena", "Hipólito", "Sebastião", "Otília", "Florisbela",
	"Geraldo", "Antônia", "Augusto", "Benedita", "Saturnino",
	"Esmeralda", "Donato", "Vitória", "Romualdo", "Aurora",
	"Pedrina", "Frederico", "Idalina", "Camilo", "Margarida",
}

// nomeReal — devolve o i-ésimo nome real, com wrap e sufixo se passou do catálogo.
func nomeReal(i int) string {
	base := nomesReaisCatalogo[i%len(nomesReaisCatalogo)]
	ciclo := i / len(nomesReaisCatalogo)
	if ciclo == 0 {
		return base
	}
	return base + " " + itoa(ciclo+1)
}

// BuildProfessores — distribui NumProfessores entre NumMaterias de forma
// balanceada (round-robin). Garante pelo menos 1 prof por matéria.
// Sempre preenche os dois nomes (código curto + nome real); o frontend
// decide qual exibir.
func BuildProfessores(numProf, numMat int) []Professor {
	if numMat < 1 {
		numMat = 1
	}
	if numMat > len(materiasCatalogo) {
		numMat = len(materiasCatalogo)
	}
	if numProf < numMat {
		numProf = numMat
	}
	out := make([]Professor, numProf)
	for i := 0; i < numProf; i++ {
		m := i % numMat
		out[i] = Professor{
			ID:       i,
			Nome:     professorNome(i),
			NomeReal: nomeReal(i),
			Materia:  m,
			MatNome:  materiasCatalogo[m],
		}
	}
	return out
}

func professorNome(i int) string {
	// nomes determinísticos curtos pra UI: P01, P02, ...
	if i < 9 {
		return "P0" + string(rune('1'+i))
	}
	// P10..Pnn — bem o que cabe na célula da matriz
	return "P" + itoa(i+1)
}

func itoa(n int) string {
	if n == 0 {
		return "0"
	}
	buf := [16]byte{}
	i := len(buf)
	neg := false
	if n < 0 {
		neg = true
		n = -n
	}
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// =============================================================================
// Avaliação (fitness)
// =============================================================================

// avaliar calcula fitness + diagnostics. O slide sugere:
//
//   + bônus por aulas encadeadas (mesma matéria em até 2 horários consecutivos)
//   − penalidade por choque (mesmo prof em mais de uma turma no mesmo slot)
//   − (extra didático) penalidade por turma sem cobertura completa de matérias
//
// Maximizamos fitness.
func avaliar(matriz []int, profs []Professor, cfg Config) (fit float64, choques, bonus, faltando int) {
	slots := cfg.DiasDaSemana * cfg.AulasPorDia
	turmas := cfg.NumTurmas

	// 1) Choques: para cada slot, contar quantas vezes cada prof aparece em
	//    turmas distintas. Mais de uma ocorrência → cada excedente é um choque.
	for s := 0; s < slots; s++ {
		seen := make(map[int]int, turmas)
		for t := 0; t < turmas; t++ {
			p := matriz[s*turmas+t]
			seen[p]++
		}
		for _, c := range seen {
			if c > 1 {
				choques += c - 1
			}
		}
	}

	// 2) Bônus por aula encadeada (mesma matéria, mesmo dia, slots consecutivos,
	//    no máximo 2 horários — pareamento simples não-sobreposto).
	for t := 0; t < turmas; t++ {
		for d := 0; d < cfg.DiasDaSemana; d++ {
			a := 0
			for a < cfg.AulasPorDia-1 {
				s1 := d*cfg.AulasPorDia + a
				s2 := s1 + 1
				p1 := matriz[s1*turmas+t]
				p2 := matriz[s2*turmas+t]
				if p1 < len(profs) && p2 < len(profs) && profs[p1].Materia == profs[p2].Materia {
					bonus++
					a += 2 // não usa o slot s2 pra outro par
					continue
				}
				a++
			}
		}
	}

	// 3) Variedade: cada matéria deveria aparecer pelo menos 1× em cada turma.
	for t := 0; t < turmas; t++ {
		cobertas := make(map[int]struct{}, cfg.NumMaterias)
		for s := 0; s < slots; s++ {
			p := matriz[s*turmas+t]
			if p < len(profs) {
				cobertas[profs[p].Materia] = struct{}{}
			}
		}
		falt := cfg.NumMaterias - len(cobertas)
		if falt > 0 {
			faltando += falt
		}
	}

	fit = float64(bonus)*cfg.BonusGeminada -
		float64(choques)*cfg.PenChoque -
		float64(faltando)*cfg.PenSemVariedade
	return
}

// =============================================================================
// População inicial e operadores
// =============================================================================

func gerarPopulacaoInicial(rng *rand.Rand, profs []Professor, cfg Config) []Individuo {
	slots := cfg.DiasDaSemana * cfg.AulasPorDia
	tam := slots * cfg.NumTurmas
	pop := make([]Individuo, cfg.PopSize)
	for i := range pop {
		m := make([]int, tam)
		for k := range m {
			m[k] = rng.Intn(cfg.NumProfessores)
		}
		pop[i] = Individuo{Matriz: m}
		pop[i].Fitness, pop[i].Choques, pop[i].Bonus, pop[i].Faltando = avaliar(m, profs, cfg)
	}
	return pop
}

// selecionarTorneio — k sorteados sem reposição, devolve os 2 mais aptos.
// Maximizamos fitness, então comparação é por > (maior é melhor).
func selecionarTorneio(pop []Individuo, k int, rng *rand.Rand) (Individuo, Individuo) {
	if k > len(pop) {
		k = len(pop)
	}
	if k < 2 {
		k = 2
	}
	perm := rng.Perm(len(pop))[:k]
	sort.Slice(perm, func(i, j int) bool {
		return pop[perm[i]].Fitness > pop[perm[j]].Fitness
	})
	return clonarIndividuo(pop[perm[0]]), clonarIndividuo(pop[perm[1]])
}

// cruzamentoPorLinhas — o operador-chave do slide. Para cada LINHA (slot),
// herda a linha inteira de um dos pais. Se ProbCruzamento < 1, em alguns
// casais o filho é cópia direta de um pai (sem mistura).
//
// O cromossomo é flat por slot, então "linha" = bloco contíguo de
// NumTurmas inteiros. Trocar a linha = copiar esse bloco do pai escolhido.
func cruzamentoPorLinhas(p1, p2 Individuo, cfg Config, rng *rand.Rand) (Individuo, Individuo) {
	slots := cfg.DiasDaSemana * cfg.AulasPorDia
	t := cfg.NumTurmas
	f1 := make([]int, slots*t)
	f2 := make([]int, slots*t)
	for s := 0; s < slots; s++ {
		from1, from2 := p1.Matriz, p2.Matriz
		if rng.Float64() < 0.5 {
			from1, from2 = from2, from1
		}
		copy(f1[s*t:(s+1)*t], from1[s*t:(s+1)*t])
		copy(f2[s*t:(s+1)*t], from2[s*t:(s+1)*t])
	}
	return Individuo{Matriz: f1}, Individuo{Matriz: f2}
}

// mutacaoCelula — sortei célula a célula e, com prob ProbMutacao, atribui
// um professor aleatório (uniforme). Equivalente a "flip" da aula 10/11,
// mas no espaço de inteiros [0, NumProf-1].
func mutacaoCelula(ind *Individuo, cfg Config, rng *rand.Rand) {
	for k := range ind.Matriz {
		if rng.Float64() < cfg.ProbMutacao {
			ind.Matriz[k] = rng.Intn(cfg.NumProfessores)
		}
	}
}

// =============================================================================
// Helpers
// =============================================================================

func clonarIndividuo(src Individuo) Individuo {
	m := make([]int, len(src.Matriz))
	copy(m, src.Matriz)
	return Individuo{
		Matriz:   m,
		Fitness:  src.Fitness,
		Choques:  src.Choques,
		Bonus:    src.Bonus,
		Faltando: src.Faltando,
	}
}

func extrairElites(pop []Individuo, p int) []Individuo {
	if p <= 0 {
		return nil
	}
	if p > len(pop) {
		p = len(pop)
	}
	idxs := make([]int, len(pop))
	for i := range idxs {
		idxs[i] = i
	}
	sort.Slice(idxs, func(i, j int) bool {
		return pop[idxs[i]].Fitness > pop[idxs[j]].Fitness
	})
	elites := make([]Individuo, p)
	for i := 0; i < p; i++ {
		elites[i] = clonarIndividuo(pop[idxs[i]])
	}
	return elites
}

// =============================================================================
// Treinar — orquestra o AG, emite Step por geração via canal.
// =============================================================================

func Treinar(progressCh chan<- Step, cfg Config) Result {
	cfg = sanitizar(cfg)
	seed := cfg.Seed
	if seed == 0 {
		seed = rand.Int63()
	}
	rng := rand.New(rand.NewSource(seed))

	profs := BuildProfessores(cfg.NumProfessores, cfg.NumMaterias)
	pop := gerarPopulacaoInicial(rng, profs, cfg)

	histMelhor := make([]float64, 0, cfg.MaxGeracoes)
	histMedia := make([]float64, 0, cfg.MaxGeracoes)
	histChoq := make([]int, 0, cfg.MaxGeracoes)
	histBon := make([]int, 0, cfg.MaxGeracoes)

	melhorGlobal := Individuo{Fitness: -1e18}

	for g := 0; g < cfg.MaxGeracoes; g++ {
		// estatísticas
		melhorIdx := 0
		soma := 0.0
		for i, ind := range pop {
			soma += ind.Fitness
			if ind.Fitness > pop[melhorIdx].Fitness {
				melhorIdx = i
			}
		}
		melhor := pop[melhorIdx]
		media := soma / float64(len(pop))
		histMelhor = append(histMelhor, melhor.Fitness)
		histMedia = append(histMedia, media)
		histChoq = append(histChoq, melhor.Choques)
		histBon = append(histBon, melhor.Bonus)

		if melhor.Fitness > melhorGlobal.Fitness {
			melhorGlobal = clonarIndividuo(melhor)
		}

		if progressCh != nil {
			progressCh <- Step{
				Geracao:     g + 1,
				MelhorFit:   melhor.Fitness,
				MelhorIndiv: clonarIndividuo(melhor),
				MediaFit:    media,
				Choques:     melhor.Choques,
				Bonus:       melhor.Bonus,
				Faltando:    melhor.Faltando,
			}
		}

		// próxima geração
		elites := extrairElites(pop, cfg.Elitismo)
		precisamos := cfg.PopSize - len(elites)
		numCasais := (precisamos + 1) / 2

		filhos := make([]Individuo, 0, 2*numCasais)
		for c := 0; c < numCasais; c++ {
			paiA, paiB := selecionarTorneio(pop, cfg.TamanhoTorneio, rng)
			var f1, f2 Individuo
			if rng.Float64() < cfg.ProbCruzamento {
				f1, f2 = cruzamentoPorLinhas(paiA, paiB, cfg, rng)
			} else {
				f1 = clonarIndividuo(paiA)
				f2 = clonarIndividuo(paiB)
			}
			mutacaoCelula(&f1, cfg, rng)
			mutacaoCelula(&f2, cfg, rng)
			f1.Fitness, f1.Choques, f1.Bonus, f1.Faltando = avaliar(f1.Matriz, profs, cfg)
			f2.Fitness, f2.Choques, f2.Bonus, f2.Faltando = avaliar(f2.Matriz, profs, cfg)
			filhos = append(filhos, f1, f2)
		}
		if len(filhos) > precisamos {
			filhos = filhos[:precisamos]
		}
		pop = append(elites, filhos...)
	}

	return Result{
		Geracoes:    cfg.MaxGeracoes,
		MelhorIndiv: melhorGlobal,
		HistMelhor:  histMelhor,
		HistMedia:   histMedia,
		HistChoques: histChoq,
		HistBonus:   histBon,
		Professores: profs,
		Cfg:         cfg,
	}
}

func sanitizar(cfg Config) Config {
	if cfg.NumProfessores < 2 {
		cfg.NumProfessores = 2
	}
	if cfg.NumTurmas < 1 {
		cfg.NumTurmas = 1
	}
	if cfg.AulasPorDia < 1 {
		cfg.AulasPorDia = 1
	}
	if cfg.DiasDaSemana < 1 {
		cfg.DiasDaSemana = 1
	}
	if cfg.NumMaterias < 1 {
		cfg.NumMaterias = 1
	}
	if cfg.NumMaterias > cfg.NumProfessores {
		cfg.NumMaterias = cfg.NumProfessores
	}
	if cfg.NumMaterias > len(materiasCatalogo) {
		cfg.NumMaterias = len(materiasCatalogo)
	}
	if cfg.PopSize < 4 {
		cfg.PopSize = 4
	}
	if cfg.PopSize%2 != 0 {
		cfg.PopSize++
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
	if cfg.TamanhoTorneio < 2 {
		cfg.TamanhoTorneio = 2
	}
	if cfg.TamanhoTorneio > cfg.PopSize {
		cfg.TamanhoTorneio = cfg.PopSize
	}
	if cfg.Elitismo < 0 {
		cfg.Elitismo = 0
	}
	if cfg.Elitismo >= cfg.PopSize {
		cfg.Elitismo = cfg.PopSize - 2
	}
	if cfg.BonusGeminada < 0 {
		cfg.BonusGeminada = 0
	}
	if cfg.PenChoque < 0 {
		cfg.PenChoque = 0
	}
	if cfg.PenSemVariedade < 0 {
		cfg.PenSemVariedade = 0
	}
	return cfg
}
