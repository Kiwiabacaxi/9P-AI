# EMNIST Letters Dataset

Dataset derivado do [NIST Special Database 19](https://www.nist.gov/srd/nist-special-database-19), reformatado para 28x28 grayscale (mesmo formato do MNIST).

## Download

Baixe os 4 arquivos do EMNIST Letters split:

```bash
# Download do NIST (todas as splits, ~536MB)
curl -LO https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip
# Extrair apenas o split "letters":
unzip -jo gzip.zip "gzip/emnist-letters-*" -d .
rm gzip.zip
```

## Formato

- **Imagens**: IDX3 (magic 2051), 28x28 pixels, grayscale 0-255
- **Labels**: IDX1 (magic 2049), valores 1-26 (A=1, Z=26)
- **Treino**: ~88,800 imagens
- **Teste**: ~14,800 imagens
- **Classes**: 26 (A-Z)

## Referência

Cohen, G., Afshar, S., Tapson, J., & van Schaik, A. (2017). EMNIST: an extension of MNIST to handwritten letters.
