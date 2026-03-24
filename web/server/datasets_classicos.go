package main

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

func portasLogicas() []PortaLogica {
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

// Padrões 7×7 para Perceptron Letras (A e B)
const percNLinhas = 7
const percNColunas = 7
const percNIn = 49 // 7×7

func percLetraA() [percNIn]float64 {
	grade := [percNLinhas][percNColunas]int{
		{-1, 1, 1, 1, 1, 1, -1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
	}
	var out [percNIn]float64
	idx := 0
	for _, row := range grade {
		for _, v := range row {
			out[idx] = float64(v)
			idx++
		}
	}
	return out
}

func percLetraB() [percNIn]float64 {
	grade := [percNLinhas][percNColunas]int{
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
	}
	var out [percNIn]float64
	idx := 0
	for _, row := range grade {
		for _, v := range row {
			out[idx] = float64(v)
			idx++
		}
	}
	return out
}

// Padrões 5×7 para MADALINE (A–M)
const madNLinhas = 7
const madNColunas = 5
const madNIn = 35
const madNLetras = 13

var madNomes = [madNLetras]string{"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M"}

func madDataset() [madNLetras][madNIn]int {
	grade := [madNLetras][madNLinhas][madNColunas]int{
		// A
		{{-1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// B
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}},
		// C
		{{-1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {-1, 1, 1, 1, 1}},
		// D
		{{1, 1, 1, 1, -1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, -1}},
		// E
		{{1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}},
		// F
		{{1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}},
		// G
		{{-1, 1, 1, 1, 1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, 1}},
		// H
		{{1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, 1, 1, 1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
		// I
		{{1, 1, 1, 1, 1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {-1, -1, 1, -1, -1}, {1, 1, 1, 1, 1}},
		// J
		{{-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {-1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {-1, 1, 1, 1, -1}},
		// K
		{{1, -1, -1, -1, 1}, {1, -1, -1, 1, -1}, {1, -1, 1, -1, -1}, {1, 1, -1, -1, -1}, {1, -1, 1, -1, -1}, {1, -1, -1, 1, -1}, {1, -1, -1, -1, 1}},
		// L
		{{1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, -1, -1, -1, -1}, {1, 1, 1, 1, 1}},
		// M
		{{1, -1, -1, -1, 1}, {1, 1, -1, 1, 1}, {1, -1, 1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}, {1, -1, -1, -1, 1}},
	}
	var dataset [madNLetras][madNIn]int
	for l := 0; l < madNLetras; l++ {
		idx := 0
		for i := 0; i < madNLinhas; i++ {
			for j := 0; j < madNColunas; j++ {
				dataset[l][idx] = grade[l][i][j]
				idx++
			}
		}
	}
	return dataset
}
