package perceptronletras

// =============================================================================
// Padrões 7×7 para Perceptron Letras (A e B)
// =============================================================================

const percNLinhas = 7
const percNColunas = 7
const NIn = 49 // 7×7

func LetraA() [NIn]float64 {
	grade := [percNLinhas][percNColunas]int{
		{-1, 1, 1, 1, 1, 1, -1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, 1, 1, 1, 1, 1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
		{1, -1, -1, -1, -1, -1, 1},
	}
	var out [NIn]float64
	idx := 0
	for _, row := range grade {
		for _, v := range row {
			out[idx] = float64(v)
			idx++
		}
	}
	return out
}

func LetraB() [NIn]float64 {
	grade := [percNLinhas][percNColunas]int{
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, -1, -1, -1, -1, 1, -1},
		{1, 1, 1, 1, 1, -1, -1},
	}
	var out [NIn]float64
	idx := 0
	for _, row := range grade {
		for _, v := range row {
			out[idx] = float64(v)
			idx++
		}
	}
	return out
}
