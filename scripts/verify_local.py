"""Verify local weights, GPU and both generators. Never calls a language/search API."""

import argparse, csv, datetime, importlib.metadata, json, os
from pathlib import Path
import subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from agentD.runtime_config import load_runtime, reinvent_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    parser.add_argument("--skip-sampling", action="store_true")
    args = parser.parse_args()
    import torch
    import numpy as np
    import faiss
    from rdkit import Chem
    from agentD.local_embeddings import get_embeddings

    cfg = load_runtime()
    work = (
        ROOT
        / "reports"
        / ("offline-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    )
    work.mkdir(parents=True)
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "device": args.device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "live_api_requests": 0,
        "sampling": {},
    }
    if args.device == "cuda:0":
        assert torch.cuda.is_available(), "CUDA device unavailable"
        torch.testing.assert_close(
            (torch.ones(4, device="cuda") * 2).cpu(), torch.full((4,), 2.0)
        )
        report["gpu"] = torch.cuda.get_device_name(0)
    embed = get_embeddings()
    matrix = np.asarray(
        embed.embed_documents(
            ["BCL-2 inhibition triggers apoptosis.", "Rain and snow weather forecast."]
        ),
        dtype="float32",
    )
    assert matrix.shape == (2, 384)
    index = faiss.IndexFlatL2(embed.dimension)
    index.add(matrix)
    _, ids = index.search(
        np.asarray([embed.embed_query("白血病 BCL-2 细胞凋亡")], dtype="float32"), 1
    )
    assert ids[0, 0] == 0, "Multilingual source retrieval failed"
    report["embedding_dimension"] = embed.dimension
    seed = work / "seed.smi"
    seed.write_text("CC(=O)Oc1ccccc1C(=O)O\n")
    if not args.skip_sampling:
        for label, prior in [
            ("Reinvent", "reinvent.prior"),
            ("Mol2Mol", "mol2mol_medium_similarity.prior"),
        ]:
            output = work / (label + ".csv")
            config = work / (label + ".toml")
            log = work / (label + ".log")
            text = f'run_type="sampling"\ndevice={json.dumps(args.device)}\n[parameters]\nmodel_file={json.dumps((cfg.reinvent_path / "priors" / prior).as_posix())}\noutput_file={json.dumps(output.as_posix())}\nnum_smiles=2\nunique_molecules=true\nrandomize_smiles=true\n'
            if label == "Mol2Mol":
                text += f'smiles_file={json.dumps(seed.as_posix())}\nsample_strategy="multinomial"\n'
            config.write_text(text, encoding="utf-8")
            with (work / (label + ".console.log")).open(
                "w", encoding="utf-8"
            ) as stream:
                subprocess.run(
                    reinvent_command(config, log),
                    cwd=work,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            with output.open(encoding="utf-8-sig") as stream:
                reader = csv.DictReader(stream)
                key = next(k for k in reader.fieldnames if k.lower() == "smiles")
                smiles = [r[key] for r in reader]
            assert smiles and all(
                s and Chem.MolFromSmiles(s) is not None for s in smiles
            ), f"{label}: invalid/empty SMILES"
            report["sampling"][label] = {
                "valid_count": len(smiles),
                "output": str(output),
            }
            print(f"{label}: {len(smiles)} valid SMILES", flush=True)
    report["packages"] = {
        name: importlib.metadata.version(name)
        for name in [
            "agentD",
            "reinvent",
            "torch",
            "torchvision",
            "faiss" if os.name == "nt" else "faiss-cpu",
            "transformers",
            "langchain",
            "langchain-openai",
            "numpy",
        ]
    }
    (work / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
