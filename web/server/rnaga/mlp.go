package rnaga

import (
	"fmt"
	"hash/fnv"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// =============================================================================
// MLP do Trabalho 15 — rede tanh em TODAS as camadas, com matrizes gonum.
//
// O AG define a arquitetura (camadas/neurônios) e os hiperparâmetros; aqui a
// rede é treinada (online ou offline/batch) e devolvemos o MSE em UNIDADES REAIS
// (denormalizando a saída quando o cromossomo pede normalização) — assim a
// comparação entre indivíduos é justa e arquiteturas sem normalização ficam
// claramente ruins (tanh satura em ±1 vs. alvos ~58–312).
//
// No modo offline, o forward/backward processa os 100 padrões como uma matriz
// 100×15 de uma vez — é onde o BLAS do gonum (C.Mul) realmente acelera.
// =============================================================================

const (
	entradaMin, entradaMax = 3.0, 1457.0
	saidaMin, saidaMax     = 58.0, 312.0
	numEntradas            = 15
	numSaidas              = 13
)

func normaliza(v, lo, hi float64) float64   { return 2*(v-lo)/(hi-lo) - 1 }
func denormaliza(v, lo, hi float64) float64 { return (v+1)/2*(hi-lo) + lo }

// rede — pesos por camada em matrizes gonum; bias em vetores.
type rede struct {
	sizes []int
	W     []*mat.Dense // W[l]: sizes[l] × sizes[l+1]
	B     [][]float64  // B[l]: len sizes[l+1]
}

func novaRede(sizes []int, rng *rand.Rand) *rede {
	r := &rede{sizes: sizes}
	for l := 0; l+1 < len(sizes); l++ {
		in, out := sizes[l], sizes[l+1]
		data := make([]float64, in*out)
		lim := 1.0 / math.Sqrt(float64(in)) // init tipo Xavier
		for i := range data {
			data[i] = (rng.Float64()*2 - 1) * lim
		}
		r.W = append(r.W, mat.NewDense(in, out, data))
		r.B = append(r.B, make([]float64, out))
	}
	return r
}

// forward — propaga o batch A (P×sizes[0]); devolve as ativações de cada camada
// (acts[0] = entrada, acts[L] = saída). tanh em todas as camadas.
func (r *rede) forward(A *mat.Dense) []*mat.Dense {
	acts := make([]*mat.Dense, len(r.sizes))
	acts[0] = A
	for l := 0; l < len(r.W); l++ {
		var z mat.Dense
		z.Mul(acts[l], r.W[l]) // BLAS dgemm
		zr := z.RawMatrix()
		out := r.sizes[l+1]
		for i := 0; i < zr.Rows; i++ {
			off := i * zr.Stride
			for j := 0; j < out; j++ {
				zr.Data[off+j] = math.Tanh(zr.Data[off+j] + r.B[l][j])
			}
		}
		acts[l+1] = &z
	}
	return acts
}

// backward — um passo de gradiente (batch) com alvo T (P×numSaidas) no espaço de
// treino. Atualiza pesos/bias in-place. tanh' = 1 − a².
func (r *rede) backward(acts []*mat.Dense, T *mat.Dense, lr float64) {
	L := len(r.W)
	P, _ := acts[L].Dims()
	scale := lr / float64(P)

	// dZ na saída: (A_L − T) ⊙ (1 − A_L²)
	var dZ mat.Dense
	dZ.Sub(acts[L], T)
	{
		dr := dZ.RawMatrix()
		ar := acts[L].RawMatrix()
		for k := range dr.Data {
			a := ar.Data[k]
			dr.Data[k] *= 1 - a*a
		}
	}
	dz := &dZ

	for l := L - 1; l >= 0; l-- {
		// gradientes com os pesos ATUAIS (antes de propagar p/ trás)
		var dW mat.Dense
		dW.Mul(acts[l].T(), dz) // (in×P)·(P×out)
		var dA mat.Dense
		dA.Mul(dz, r.W[l].T()) // (P×out)·(out×in) — usa W[l] antes do update

		// update W[l]
		wr := r.W[l].RawMatrix()
		gr := dW.RawMatrix()
		for k := range wr.Data {
			wr.Data[k] -= scale * gr.Data[k]
		}
		// update B[l] (soma das linhas de dz)
		zr := dz.RawMatrix()
		out := r.sizes[l+1]
		for j := 0; j < out; j++ {
			s := 0.0
			for i := 0; i < zr.Rows; i++ {
				s += zr.Data[i*zr.Stride+j]
			}
			r.B[l][j] -= scale * s
		}

		// dz da camada anterior: dA ⊙ (1 − acts[l]²)
		if l > 0 {
			dar := dA.RawMatrix()
			ar := acts[l].RawMatrix()
			for k := range dar.Data {
				a := ar.Data[k]
				dar.Data[k] *= 1 - a*a
			}
		}
		dz = &dA
	}
}

// treinarOnline — backprop estocástico (atualiza pesos a cada padrão) operando
// direto sobre os dados das matrizes gonum (RawMatrix), com buffers de ativação
// e delta pré-alocados → zero alocação dentro do loop de épocas.
//
// realoca=true reproduz o jeito "ingênuo": realoca os buffers a CADA padrão (a
// matemática é idêntica → mesmo resultado, mas a pressão de GC custa caro). É só
// pro benchmark; em produção usamos realoca=false (buffers reaproveitados).
func (r *rede) treinarOnline(X, T [][]float64, epocas int, lr float64, rng *rand.Rand, realoca bool) {
	L := len(r.W)
	var a, delta [][]float64
	alocar := func() {
		a = make([][]float64, L+1)
		delta = make([][]float64, L+1)
		for l := range r.sizes {
			a[l] = make([]float64, r.sizes[l])
			delta[l] = make([]float64, r.sizes[l])
		}
	}
	if !realoca {
		alocar()
	}
	P := len(X)
	ordem := make([]int, P)
	for i := range ordem {
		ordem[i] = i
	}
	for e := 0; e < epocas; e++ {
		rng.Shuffle(P, func(i, j int) { ordem[i], ordem[j] = ordem[j], ordem[i] })
		for _, idx := range ordem {
			if realoca {
				alocar()
			}
			copy(a[0], X[idx])
			// forward
			for l := 0; l < L; l++ {
				wr := r.W[l].RawMatrix()
				in, out := r.sizes[l], r.sizes[l+1]
				for j := 0; j < out; j++ {
					z := r.B[l][j]
					for i := 0; i < in; i++ {
						z += a[l][i] * wr.Data[i*wr.Stride+j]
					}
					a[l+1][j] = math.Tanh(z)
				}
			}
			// delta da saída
			for k := 0; k < r.sizes[L]; k++ {
				o := a[L][k]
				delta[L][k] = (o - T[idx][k]) * (1 - o*o)
			}
			// backward + update (delta da camada anterior usa pesos ANTIGOS)
			for l := L - 1; l >= 0; l-- {
				wr := r.W[l].RawMatrix()
				in, out := r.sizes[l], r.sizes[l+1]
				if l > 0 {
					for i := 0; i < in; i++ {
						s := 0.0
						for j := 0; j < out; j++ {
							s += delta[l+1][j] * wr.Data[i*wr.Stride+j]
						}
						delta[l][i] = s * (1 - a[l][i]*a[l][i])
					}
				}
				for i := 0; i < in; i++ {
					ai := a[l][i]
					base := i * wr.Stride
					for j := 0; j < out; j++ {
						wr.Data[base+j] -= lr * ai * delta[l+1][j]
					}
				}
				for j := 0; j < out; j++ {
					r.B[l][j] -= lr * delta[l+1][j]
				}
			}
		}
	}
}

// arquiteturaDe — sizes [15, h, h, ..., h, 13] conforme o cromossomo.
func arquiteturaDe(c Cromossomo) []int {
	h, nc := c.Neuronios(), c.Camadas()
	sizes := make([]int, 0, nc+2)
	sizes = append(sizes, numEntradas)
	for i := 0; i < nc; i++ {
		sizes = append(sizes, h)
	}
	sizes = append(sizes, numSaidas)
	return sizes
}

// chaveCanonica — identidade da arquitetura EFETIVA (decodificada). Mesma chave ⇒
// mesma semente de pesos ⇒ mesmo MSE ⇒ memoização válida e execução reprodutível.
func chaveCanonica(c Cromossomo, teto int) string {
	on, nm := 2, 2
	if c.Online() {
		on = 1
	}
	if c.Normaliza() {
		nm = 1
	}
	return fmt.Sprintf("%d|%d|%.6f|%d|%d|%d",
		c.Neuronios(), c.Camadas(), c.TaxaAprend(), c.epocasEfetivas(teto), on, nm)
}

// AvaliarMSE — treina a rede definida pelo cromossomo e devolve o MSE em unidades
// reais (58–312). Determinístico para (chaveCanonica, seedRun).
func AvaliarMSE(c Cromossomo, ds Dataset, teto int, seedRun int64) float64 {
	return avaliarMSEOpts(c, ds, teto, seedRun, false)
}

// avaliarMSEOpts — como AvaliarMSE, mas com o toggle onlineRealoca (benchmark).
func avaliarMSEOpts(c Cromossomo, ds Dataset, teto int, seedRun int64, onlineRealoca bool) float64 {
	sizes := arquiteturaDe(c)
	h := fnv.New64a()
	h.Write([]byte(chaveCanonica(c, teto)))
	seed := int64(h.Sum64()) ^ seedRun
	rng := rand.New(rand.NewSource(seed))
	r := novaRede(sizes, rng)

	normalizar := c.Normaliza()
	epocas := c.epocasEfetivas(teto)
	online := c.Online()
	lr := c.TaxaAprend()
	P := len(ds.X)

	// matrizes de entrada (X) e alvo de treino (T) no espaço de treino
	Xdata := make([]float64, P*numEntradas)
	Tdata := make([]float64, P*numSaidas)
	for i := 0; i < P; i++ {
		for j := 0; j < numEntradas; j++ {
			v := ds.X[i][j]
			if normalizar {
				v = normaliza(v, entradaMin, entradaMax)
			}
			Xdata[i*numEntradas+j] = v
		}
		for k := 0; k < numSaidas; k++ {
			v := ds.Y[i][k]
			if normalizar {
				v = normaliza(v, saidaMin, saidaMax)
			}
			Tdata[i*numSaidas+k] = v
		}
	}
	X := mat.NewDense(P, numEntradas, Xdata)
	T := mat.NewDense(P, numSaidas, Tdata)

	if online {
		// Modo online (1 padrão por vez): caminho de slice puro com buffers
		// pré-alocados — matriz 1×15 não se beneficia do BLAS e a alocação por
		// padrão (gonum) dominaria o custo. O gonum fica no offline (batch).
		Xrows := make([][]float64, P)
		Trows := make([][]float64, P)
		for i := 0; i < P; i++ {
			Xrows[i] = Xdata[i*numEntradas : (i+1)*numEntradas]
			Trows[i] = Tdata[i*numSaidas : (i+1)*numSaidas]
		}
		r.treinarOnline(Xrows, Trows, epocas, lr, rng, onlineRealoca)
	} else {
		for e := 0; e < epocas; e++ {
			acts := r.forward(X)
			r.backward(acts, T, lr)
		}
	}

	// MSE em unidades reais
	out := r.forward(X)[len(sizes)-1]
	mse := 0.0
	for i := 0; i < P; i++ {
		for k := 0; k < numSaidas; k++ {
			o := out.At(i, k)
			pred := o
			if normalizar {
				pred = denormaliza(o, saidaMin, saidaMax)
			}
			d := ds.Y[i][k] - pred
			mse += d * d
		}
	}
	mse /= float64(P * numSaidas)
	if math.IsNaN(mse) || math.IsInf(mse, 0) {
		mse = 1e18 // arquitetura explodiu → fitness péssimo
	}
	return mse
}
