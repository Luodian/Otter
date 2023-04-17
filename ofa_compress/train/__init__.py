from .ofa_distillers import BasicTrainer, OFADistiller
from textbrewer.distillers import BasicDistiller,GeneralDistiller,MultiTaskDistiller,MultiTeacherDistiller
from .ofa_configurations import OFADistillationConfig
from textbrewer.configurations import TrainingConfig


from .ofa_presets import FEATURES
from .ofa_presets import ADAPTOR_KEYS
from .ofa_presets import KD_LOSS_MAP, MATCH_LOSS_MAP, PROJ_MAP
from .ofa_presets import WEIGHT_SCHEDULER, TEMPERATURE_SCHEDULER
from .ofa_presets import register_new

Distillers = {
    'Basic': BasicDistiller,
    'General': GeneralDistiller,
    'MultiTeacher': MultiTeacherDistiller,
    'MultiTask': MultiTaskDistiller,
    'Train': BasicTrainer,
    'OFA': OFADistiller
}

