# Redes Neurais Artificiais

Projeto da disciplina de **Inteligencia Artificial** na faculdade, focado em dar uma visualizacao bonita e interativa para os algoritmos classicos de redes neurais.

A ideia e acompanhar a historia da area: comecamos com modelos simples como Hebb (literalmente uma regra de uma linha), passamos pelo Perceptron, MADALINE, ate chegar no MLP com backpropagation. Algumas implementacoes sao propositalmente "ingenuas" — com 3 for aninhados e tudo mais — porque o objetivo e entender o algoritmo pela descoberta, nao otimizar performance.

> **[Abrir no browser](https://kiwiabacaxi.github.io/9P-AI/)** — roda direto via WebAssembly, sem instalar nada.

## Versao web vs local

O projeto tem uma **versao web** que roda no browser e uma **versao local** com servidor Go.

A versao web funciona bem pra dar uma olhada rapida, mas rodar localmente tem vantagens: o backend em Go e mais rapido, suporta todos os backends de otimizacao (incluindo o matricial com BLAS), e o streaming de progresso via SSE e mais responsivo.

```bash
# rodar local (recomendado)
cd web
make run

# ou so abrir no browser
# https://kiwiabacaxi.github.io/9P-AI/
```

Alem da versao web, cada algoritmo tambem tem uma **TUI interativa** no terminal com animacoes, slides explicativos e modos de teste — veja a [pasta cli](cli/README.md).

## O que tem aqui

Seguindo a ordem das aulas:

| Aula | Algoritmo | O que faz |
|------|-----------|-----------|
| 02 | **Hebb** | Portas logicas (AND, OR...) com a regra mais simples que existe |
| 03 | **Perceptron** | Reconhecimento de letras A/B (7x7 pixels) e portas logicas |
| 04 | **MADALINE** | Reconhecimento de letras A-M com 13 ADALINEs |
| 05 | **MLP** | Backpropagation — exemplo numerico dos slides e reconhecimento A-Z |
| 05 | **Image Regression** | Uma rede MLP aprendendo a "pintar" imagens pixel por pixel |

## Estrutura

```
cli/                     TUIs interativas no terminal (Go + Charm)
  trab01-hebb/           Regra de Hebb
  trab02-perceptron-*/   Perceptron (letras e portas)
  trab03-madaline/       MADALINE
  desafio-mlp/           MLP 3->2->3
  desafio-mlp-letras/    MLP A-Z

web/                     Versao web (servidor Go + frontend)
  server/                Backend com todos os algoritmos como packages
  static/                Frontend (HTML/CSS/JS puro, sem frameworks)

slides/                  PDFs das aulas
```

## Como rodar

**Web (local):**
```bash
cd web && make run
```

**TUI no terminal:**
```bash
cd cli
go run ./trab01-hebb
go run ./desafio-mlp-letras
# etc.
```

**Web (online):** basta acessar https://kiwiabacaxi.github.io/9P-AI/

## Tecnologias

- **Go** — tudo: algoritmos, servidor HTTP, WebAssembly, TUI
- **Charm** (Bubble Tea + Lipgloss) — interfaces de terminal interativas
- **WebAssembly** — mesmos algoritmos Go rodando no browser
- **HTML/CSS/JS** — frontend sem frameworks
- **GitHub Actions** — deploy automatico no Pages
