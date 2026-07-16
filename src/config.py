import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "experiment.json"


class ExperimentConfig:
    def __init__(self, path, raw):
        self.path = path
        self.raw = raw

    @property
    def root(self):
        return self.path.resolve().parents[1]

    @property
    def datasets(self):
        return self.raw["datasets"]

    @property
    def seeds(self):
        return [int(seed) for seed in self.raw["project"]["seeds"]]

    def dataset(self, key):
        return self.datasets[key]

    def model(self, name):
        canonical = canonical_model_name(name)
        return self.raw["models"][canonical]

    def model_hparams(self, name, dataset):
        model_cfg = self.model(name)
        return {
            "batch_size": int(model_cfg["batch_size"]),
            **model_cfg["hyperparameters"][dataset],
        }


def canonical_model_name(name):
    normalized = name.lower().replace("_", "").replace("-", "")
    aliases = {"neumf": "neumf", "reviewonly": "reviewonly", "myreviewonly": "reviewonly"}
    return aliases[normalized]


def parse_csv(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv(value):
    return [int(item) for item in parse_csv(value)]


def load_config(path=None):
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return ExperimentConfig(path=config_path, raw=raw)
