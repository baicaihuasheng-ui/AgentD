"""Portable runtime settings. Credentials are never serialized with task configs."""

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str = field(default="", repr=False)
    api_base: str = "https://api.deepseek.com"
    model: str = "deepseek-flash"
    embedding_path: Path = ROOT.parent / "agentd-models" / "multilingual-e5-small"
    embedding_device: str = "cpu"
    reinvent_path: Path = ROOT.parent / "REINVENT4"
    reinvent_executable: Path = Path(sys.executable).parent / (
        "Scripts/reinvent.exe" if os.name == "nt" else "reinvent"
    )
    sampling_device: str = "cuda:0"
    protein_model_path: Path = ROOT.parent / "agentd-models" / "prot_t5_xl_half_uniref50-enc"
    molecule_model_path: Path = ROOT.parent / "agentd-models" / "MoLFormer-XL-both-10pct"
    affinity_checkpoint_path: Path = ROOT / "agentD/tools/BAPULM_results_molformer_reproduce_json.pth"
    affinity_cache_path: Path = ROOT / "runs/affinity-cache"
    affinity_device: str = "auto"
    affinity_precision: str = "float16"

    @classmethod
    def from_env(cls):
        path = (
            Path(os.getenv("AGENTD_CONFIG", ROOT / "runtime.local.json"))
            .expanduser()
            .resolve()
        )
        settings = (
            json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        )
        if any(
            k.lower()
            in {"api_key", "deepseek_api_key", "openai_api_key", "serper_api_key"}
            for k in settings
        ):
            raise ValueError("Credentials belong in DEEPSEEK_API_KEY, not runtime JSON")

        def value(name, default):
            return os.getenv("AGENTD_" + name.upper(), settings.get(name, default))

        def absolute(name, default):
            p = Path(value(name, str(default))).expanduser()
            return (path.parent / p).resolve() if not p.is_absolute() else p.resolve()

        return cls(
            api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
            api_base=value("api_base", cls.api_base).rstrip("/"),
            model=value("model", cls.model),
            embedding_path=absolute("embedding_path", cls.embedding_path),
            embedding_device=value("embedding_device", "cpu"),
            reinvent_path=absolute("reinvent_path", cls.reinvent_path),
            reinvent_executable=absolute(
                "reinvent_executable", cls.reinvent_executable
            ),
            sampling_device=value("sampling_device", "cuda:0"),
            protein_model_path=absolute("protein_model_path", cls.protein_model_path),
            molecule_model_path=absolute("molecule_model_path", cls.molecule_model_path),
            affinity_checkpoint_path=absolute("affinity_checkpoint_path", cls.affinity_checkpoint_path),
            affinity_cache_path=absolute("affinity_cache_path", cls.affinity_cache_path),
            affinity_device=value("affinity_device", "auto"),
            affinity_precision=value("affinity_precision", "float16"),
        )

    def require_key(self):
        if not self.api_key:
            raise ValueError("Set DEEPSEEK_API_KEY before making a live API request")
        return self.api_key


def load_runtime():
    envfile = os.getenv("AGENTD_ENV_FILE")
    if envfile:
        from dotenv import load_dotenv

        path = Path(envfile).expanduser().resolve()
        if not path.is_file():
            raise ValueError("AGENTD_ENV_FILE does not exist")
        load_dotenv(path, override=False)
    return RuntimeConfig.from_env()


def configure_runtime(config):
    """Called by runner before starting MCP; only whitelisted nonsecret settings."""
    for key in (
        "model",
        "embedding_path",
        "embedding_device",
        "reinvent_path",
        "reinvent_executable",
        "sampling_device",
        "protein_model_path",
        "molecule_model_path",
        "affinity_checkpoint_path",
        "affinity_cache_path",
        "affinity_device",
        "affinity_precision",
    ):
        if key in config:
            val = str(config[key])
            if key.endswith("_path") or key.endswith("_executable"):
                val = str(Path(val).expanduser().resolve())
            os.environ["AGENTD_" + key.upper()] = val
    load_runtime()


def reinvent_command(config_path, log_path):
    return [
        str(load_runtime().reinvent_executable),
        "-l",
        str(log_path),
        str(config_path),
    ]
