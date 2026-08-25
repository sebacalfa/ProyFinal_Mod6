from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "adult_raw.csv"
OUTPUT_DIR = ROOT / "resultados_pipeline"
FIGURES_DIR = OUTPUT_DIR / "graficos"
MODEL_DIR = OUTPUT_DIR / "modelo"
TARGET = "income"
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EXPECTED_COLUMNS = {
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    TARGET,
}
