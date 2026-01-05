from pathlib import Path


PASTA_PROJETO = Path(__file__).resolve().parents[2]

PASTA_DADOS = PASTA_PROJETO / "dados"

# DADOS
# Dados reais (fonte: https://www.kaggle.com/datasets/camnugent/california-housing-prices/data)
REAL_DATA = PASTA_DADOS / "housing.csv"
CLEAN_DATA = PASTA_DADOS / "clean_data.parquet"
REAL_GEO_DATA = PASTA_DADOS / "california.geojson"

# coloque abaixo o caminho para os arquivos de modelos de seu projeto
PASTA_MODELOS = PASTA_PROJETO / "modelos"

# coloque abaixo outros caminhos que você julgar necessário
PASTA_RELATORIOS = PASTA_PROJETO / "relatorios"
PASTA_IMAGENS = PASTA_RELATORIOS / "imagens"
