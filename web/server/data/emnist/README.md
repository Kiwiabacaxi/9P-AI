# EMNIST Letters Dataset

Dataset derivado do [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19), reformatado para 28x28 grayscale (mesmo formato do MNIST).

## Download

Baixe os 4 arquivos do EMNIST Letters split:

```bash
# Opção 1: GitHub mirror
curl -LO https://raw.githubusercontent.com/hosford42/EMNIST/master/emnist-letters-train-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/hosford42/EMNIST/master/emnist-letters-train-labels-idx1-ubyte.gz
curl -LO https://raw.githubusercontent.com/hosford42/EMNIST/master/emnist-letters-test-images-idx3-ubyte.gz
curl -LO https://raw.githubusercontent.com/hosford42/EMNIST/master/emnist-letters-test-labels-idx1-ubyte.gz
```

## Formato

- **Imagens**: IDX3 (magic 2051), 28x28 pixels, grayscale 0-255
- **Labels**: IDX1 (magic 2049), valores 1-26 (A=1, Z=26)
- **Treino**: ~88,800 imagens
- **Teste**: ~14,800 imagens
- **Classes**: 26 (A-Z)

## Referência

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
