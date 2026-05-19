from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "processed"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"

OUTPUT_DIR = ROOT_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PLOTS_DIR = OUTPUT_DIR / "plots"
METRICS_DIR = OUTPUT_DIR / "metrics"

IMG_SIZE = 64
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 1e-3
NUM_WORKERS = 4

MODEL_NAME = "custom_cnn"
SEED = 42