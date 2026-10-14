"""
training/config.py
==================
Configuração central do pipeline de treinamento do MindFlow AI.

Todos os hiperparâmetros em um único lugar — nenhum "magic number"
nos outros módulos. Para experimentar, altere apenas este arquivo.

Estrutura esperada:
    mindflow_extractor/          ← pacote do extrator (já pronto)
    training/                    ← este módulo
    outputs/
        features/train/          ← pares .npy + _metadata.json (Train)
        features/validation/     ← pares .npy + _metadata.json (Validation)
        features/test/           ← pares .npy + _metadata.json (Test)
        training/                ← artefatos gerados por este módulo
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Caminhos base
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent   # raiz do repositório

FEATURES_DIR_TRAIN = ROOT_DIR / "outputs" / "features" / "train"
FEATURES_DIR_VAL   = ROOT_DIR / "outputs" / "features" / "validation"
FEATURES_DIR_TEST  = ROOT_DIR / "outputs" / "features" / "test"

OUTPUT_BASE_DIR = ROOT_DIR / "outputs" / "training" / "runs"


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeatureConfig:
    """Dimensões do vetor produzido pelo mindflow_extractor."""
    vector_dim:   int = 120   # total de features por frame extraídas pelo mindflow_extractor (Early Fusion)
    # Com skip_frames=4 na extração, cada frame representa ~167ms (6fps).
    # 30 frames à 6fps = 5 segundos de contexto. Testamos usar o clipe
    # quase inteiro (58 frames, ~9.7s) e o resultado não compensou: com
    # 1 janela por clipe em vez de ~3, o volume de treino caiu 3x e o
    # modelo passou a overfitar já na 1ª época (ver docs/justificativa_
    # engajamento_isolado.md). Voltamos para 30 por ser o run validado.
    sequence_len: int = 30    # frames por janela temporal (5s)

    # PERCLOS — feature derivada (ver training/feature_engineering.py),
    # calculada em cima do EAR já extraído, sem reprocessar os vídeos.
    # Concatenada como coluna 120 (índice), o vetor de treino fica 121d
    # mesmo com vector_dim=120 continuando a descrever a saída bruta do
    # mindflow_extractor.
    perclos_window_frames: int   = 6      # ~1s ao fps efetivo real (~6fps)
    perclos_ear_threshold: float = 0.22   # mesmo threshold de piscada do extrator

    # Blocos para auditoria (índices inclusivos)
    block_A: Tuple[int, int] = (0,   31)   # 32d — AUs faciais
    block_B: Tuple[int, int] = (32,  43)   # 12d — head pose
    block_C: Tuple[int, int] = (44,  67)   # 24d — gaze
    block_D: Tuple[int, int] = (68,  83)   # 16d — olhos/EAR
    block_E: Tuple[int, int] = (84,  91)   #  8d — boca/MAR
    block_F: Tuple[int, int] = (92,  115)  # 24d — postura
    block_G: Tuple[int, int] = (116, 119)  #  4d — qualidade


# ---------------------------------------------------------------------------
# Classes  — 4 classes originais do DAiSEE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClassConfig:
    """
    Rótulos originais do DAiSEE para Engagement:
      0 = very low engagement
      1 = low engagement
      2 = high engagement
      3 = very high engagement
    """
    names:       Tuple[str, ...] = (
        "very_low_engagement",
        "low_engagement",
        "high_engagement",
        "very_high_engagement",
    )
    num_classes: int = 4


# ---------------------------------------------------------------------------
# Pré-processamento
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PreprocessConfig:
    scaler:                 str   = "standard"   # "standard"|"minmax"|"robust"
    smote_k_neighbors:      int   = 5
    smote_random_state:     int   = 42
    use_pca:                bool  = True          # OBRIGATÓRIO
    pca_variance_threshold: float = 0.95          # retém 95% da variância
    pca_random_state:       int   = 42
    # Com skip_frames=4, stride de 15 mantém 50% de overlap
    sequence_stride:        int   = 15


# ---------------------------------------------------------------------------
# Arquitetura
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """
    Arquitetura completa para dataset completo DAiSEE (~5k vídeos).
    """
    lstm_hidden_1: int   = 128
    lstm_hidden_2: int   = 64
    dropout:       float = 0.5   # aumentado 0.3→0.5 para reduzir overfitting


# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TrainConfig:
    optimizer:           str   = "adam"
    learning_rate:       float = 1e-4    # reduzido 5e-4→1e-4 para generalização melhor
    weight_decay:        float = 5e-4   # era 1e-4 — regularização mais forte
    use_lr_scheduler:    bool  = True
    lr_patience:         int   = 5
    lr_factor:           float = 0.5
    max_epochs:          int   = 100
    batch_size:          int   = 64
    early_stop_patience: int   = 15
    seed:                int   = 42
    device:              str   = "auto"
    # Se definido, treina apenas uma tarefa específica (ex: 'engagement').
    # Se None, treina no modo multi-output (todas as 4 tarefas).
    target_task:         Optional[str] = None
    # Cabeça ordinal CORAL em vez de Linear+softmax nominal.
    # Requer target_task definido (só suportado em single-task).
    ordinal:              bool = False


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvalConfig:
    primary_metric: str             = "f1_macro"
    report_metrics: Tuple[str, ...] = (
        "accuracy", "f1_macro", "f1_weighted",
        "precision_macro", "recall_macro", "confusion_matrix",
    )
    save_history: bool = True


# ---------------------------------------------------------------------------
# Config agregada
# ---------------------------------------------------------------------------

@dataclass
class MindFlowTrainingConfig:
    """
    Ponto de entrada único. Use em todos os módulos:

        from training.config import MindFlowTrainingConfig
        cfg = MindFlowTrainingConfig()
    """
    features:   FeatureConfig    = field(default_factory=FeatureConfig)
    classes:    ClassConfig      = field(default_factory=ClassConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)
    eval:       EvalConfig       = field(default_factory=EvalConfig)

    # Identificador único da execução (timestamp)
    run_id: str = field(default_factory=lambda: __import__("time").strftime("%Y-%m-%d_%H-%M-%S"))

    @property
    def output_dir(self) -> Path:
        p = OUTPUT_BASE_DIR / self.run_id
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def scaler_path(self) -> Path: return self.output_dir / "scaler.joblib"
    
    @property
    def pca_path(self) -> Path: return self.output_dir / "pca.joblib"
    
    @property
    def model_path(self) -> Path: return self.output_dir / "lstm_mindflow.pt"
    
    @property
    def metrics_path(self) -> Path: return self.output_dir / "metrics.json"
    
    @property
    def history_path(self) -> Path: return self.output_dir / "training_history.json"
    
    @property
    def report_path(self) -> Path: return self.output_dir / "classification_report.txt"
    
    @property
    def plots_dir(self) -> Path:
        p = self.output_dir / "plots"
        p.mkdir(parents=True, exist_ok=True)
        return p


DEFAULT_CFG = MindFlowTrainingConfig()


if __name__ == "__main__":
    cfg = MindFlowTrainingConfig()
    print("=== MindFlow Training Config ===")
    print(f"  vector_dim    : {cfg.features.vector_dim}")
    print(f"  sequence_len  : {cfg.features.sequence_len}")
    print(f"  num_classes   : {cfg.classes.num_classes}")
    print(f"  class_names   : {cfg.classes.names}")
    print(f"  LSTM arch     : {cfg.model.lstm_hidden_1}→{cfg.model.lstm_hidden_2}→Dense({cfg.classes.num_classes})")
    print(f"  use_pca       : {cfg.preprocess.use_pca}")
    print(f"  features/train: {FEATURES_DIR_TRAIN}")
    print(f"  run_id        : {cfg.run_id}")
    print(f"  output_dir    : {cfg.output_dir}")
    print("\n✅  Config OK")
