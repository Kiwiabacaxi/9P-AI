package cnn

// =============================================================================
// Loader EMNIST Letters — formato IDX binário (mesmo do MNIST)
//
// O EMNIST é derivado do NIST Special Database 19, reformatado para 28×28
// grayscale no formato IDX usado pelo MNIST original (LeCun et al.).
//
// Formato IDX:
//   Images: magic(0x00000803) | nImages(4B) | nRows(4B) | nCols(4B) | pixels...
//   Labels: magic(0x00000801) | nLabels(4B) | labels...
//
// Peculiaridade do EMNIST: imagens são armazenadas em column-major order
// (transpostas em relação ao MNIST). Precisamos transpor cada imagem.
//
// Labels: EMNIST Letters usa 1-26 (A=1, B=2, ..., Z=26).
// Mapeamos para 0-25 internamente.
// =============================================================================

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

const (
	ImgSize    = 28
	ImgPixels  = ImgSize * ImgSize // 784
	NumClasses = 26                // A-Z
)

var LetterNames = [NumClasses]string{
	"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
	"N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
}

// EMNISTData contém o dataset carregado e pronto para uso
type EMNISTData struct {
	TrainImages [][]float64 // [nTrain][784] — pixels normalizados [0,1]
	TrainLabels []int       // [nTrain] — 0-25 (A=0, Z=25)
	TestImages  [][]float64 // [nTest][784]
	TestLabels  []int       // [nTest]
}

// LoadEMNIST carrega o dataset EMNIST Letters de um diretório.
// Busca arquivos .gz (comprimidos) ou descomprimidos.
func LoadEMNIST(dir string) (*EMNISTData, error) {
	trainImg, err := readIDXImages(findFile(dir, "emnist-letters-train-images-idx3-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("train images: %w", err)
	}
	trainLbl, err := readIDXLabels(findFile(dir, "emnist-letters-train-labels-idx1-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("train labels: %w", err)
	}
	testImg, err := readIDXImages(findFile(dir, "emnist-letters-test-images-idx3-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("test images: %w", err)
	}
	testLbl, err := readIDXLabels(findFile(dir, "emnist-letters-test-labels-idx1-ubyte"))
	if err != nil {
		return nil, fmt.Errorf("test labels: %w", err)
	}

	return &EMNISTData{
		TrainImages: trainImg,
		TrainLabels: trainLbl,
		TestImages:  testImg,
		TestLabels:  testLbl,
	}, nil
}

// findFile tenta encontrar o arquivo com ou sem extensão .gz
func findFile(dir, baseName string) string {
	// Tentar .gz primeiro
	gz := filepath.Join(dir, baseName+".gz")
	if _, err := os.Stat(gz); err == nil {
		return gz
	}
	// Tentar sem compressão
	plain := filepath.Join(dir, baseName)
	if _, err := os.Stat(plain); err == nil {
		return plain
	}
	// Retornar o caminho .gz para gerar erro informativo
	return gz
}

// openFile abre um arquivo, descomprimindo se for .gz
func openFile(path string) (io.ReadCloser, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("não foi possível abrir %s: %w\n\nBaixe o EMNIST Letters de:\nhttps://www.nist.gov/itl/products-and-services/emnist-dataset", path, err)
	}
	if strings.HasSuffix(path, ".gz") {
		gz, err := gzip.NewReader(f)
		if err != nil {
			f.Close()
			return nil, fmt.Errorf("erro ao descomprimir %s: %w", path, err)
		}
		return gz, nil
	}
	return f, nil
}

// readIDXImages lê imagens no formato IDX e retorna como slices normalizadas [0,1].
// Cada imagem é transposta (EMNIST quirk: armazenado column-major).
func readIDXImages(path string) ([][]float64, error) {
	r, err := openFile(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	// Ler header: magic(4) + nImages(4) + nRows(4) + nCols(4)
	var header [4]int32
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("erro ao ler header de imagens: %w", err)
	}
	if header[0] != 2051 {
		return nil, fmt.Errorf("magic number inválido para imagens: %d (esperado 2051)", header[0])
	}

	nImages := int(header[1])
	nRows := int(header[2])
	nCols := int(header[3])
	pixelsPerImage := nRows * nCols

	// Ler todos os pixels raw
	rawBuf := make([]byte, pixelsPerImage)
	images := make([][]float64, nImages)

	for i := range nImages {
		if _, err := io.ReadFull(r, rawBuf); err != nil {
			return nil, fmt.Errorf("erro ao ler imagem %d: %w", i, err)
		}

		img := make([]float64, pixelsPerImage)
		// Transpor: EMNIST armazena column-major, precisamos row-major
		for row := range nRows {
			for col := range nCols {
				// Column-major: rawBuf[col*nRows + row]
				// Row-major:    img[row*nCols + col]
				img[row*nCols+col] = float64(rawBuf[col*nRows+row]) / 255.0
			}
		}
		images[i] = img
	}

	return images, nil
}

// readIDXLabels lê labels no formato IDX.
// EMNIST Letters: labels 1-26 → mapeamos para 0-25.
func readIDXLabels(path string) ([]int, error) {
	r, err := openFile(path)
	if err != nil {
		return nil, err
	}
	defer r.Close()

	// Ler header: magic(4) + nLabels(4)
	var header [2]int32
	if err := binary.Read(r, binary.BigEndian, &header); err != nil {
		return nil, fmt.Errorf("erro ao ler header de labels: %w", err)
	}
	if header[0] != 2049 {
		return nil, fmt.Errorf("magic number inválido para labels: %d (esperado 2049)", header[0])
	}

	nLabels := int(header[1])
	rawBuf := make([]byte, nLabels)
	if _, err := io.ReadFull(r, rawBuf); err != nil {
		return nil, fmt.Errorf("erro ao ler labels: %w", err)
	}

	labels := make([]int, nLabels)
	for i, b := range rawBuf {
		labels[i] = int(b) - 1 // EMNIST Letters: 1-26 → 0-25
	}
	return labels, nil
}

// ImageToTensor converte uma imagem flat (784) para tensor 3D [1][28][28]
func ImageToTensor(flat []float64) [][][]float64 {
	t := make3D(1, ImgSize, ImgSize)
	for row := range ImgSize {
		for col := range ImgSize {
			t[0][row][col] = flat[row*ImgSize+col]
		}
	}
	return t
}

// ShuffleData embaralha imagens e labels in-place (mesma permutação)
func ShuffleData(rng *rand.Rand, images [][]float64, labels []int) {
	n := len(images)
	for i := n - 1; i > 0; i-- {
		j := rng.Intn(i + 1)
		images[i], images[j] = images[j], images[i]
		labels[i], labels[j] = labels[j], labels[i]
	}
}
