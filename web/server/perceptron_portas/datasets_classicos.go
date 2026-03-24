package perceptronportas

// =============================================================================
// Datasets clássicos — compartilhados entre os algoritmos
// =============================================================================

// Tabelas verdade das portas lógicas (bipolar)
type PortaLogica struct {
	Nome    string     `json:"nome"`
	Inputs  [4][2]int  `json:"inputs"`
	Targets [4]int     `json:"targets"`
	Desc    string     `json:"desc"`
}

func PortasLogicas() []PortaLogica {
	return []PortaLogica{
		{Nome: "AND", Desc: "Retorna 1 apenas quando ambas entradas são 1",
			Inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			Targets: [4]int{1, -1, -1, -1}},
		{Nome: "OR", Desc: "Retorna 1 quando pelo menos uma entrada é 1",
			Inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			Targets: [4]int{1, 1, 1, -1}},
		{Nome: "NAND", Desc: "Negação do AND — retorna -1 apenas quando ambas são 1",
			Inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			Targets: [4]int{-1, 1, 1, 1}},
		{Nome: "NOR", Desc: "Negação do OR — retorna 1 apenas quando ambas são -1",
			Inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			Targets: [4]int{-1, -1, -1, 1}},
		{Nome: "XOR", Desc: "Retorna 1 quando as entradas são diferentes (não linearmente separável)",
			Inputs:  [4][2]int{{1, 1}, {1, -1}, {-1, 1}, {-1, -1}},
			Targets: [4]int{-1, 1, 1, -1}},
	}
}
