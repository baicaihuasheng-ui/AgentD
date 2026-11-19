import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class PipelineTests(unittest.TestCase):
    def test_progress_ignores_model_cache_directories(self):
        import run_agentd
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = root / "runs/BCL2_test"
            actual.mkdir(parents=True)
            (actual / "status.json").write_text("{}")
            (root / "runs/hf-modules").mkdir()
            with patch.object(run_agentd, "__file__", str(root / "run_agentd.py")):
                self.assertEqual(Path(run_agentd.find_latest_run_dir()), actual)

    def test_generated_sampling_config_comments_unsupported_sampling_log_field(self):
        import tomllib
        from types import SimpleNamespace
        import agentD.tools.generation as generation
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "configs/toml").mkdir(parents=True)
            (root / "configs/toml/sampling.toml").write_text('run_type="sampling"')
            prior = root / "test.prior"
            prior.write_bytes(b"placeholder")
            generated = 'run_type="sampling"\ndevice="cpu"\n[parameters]\nmodel_file="' + prior.as_posix() + '"\nnum_smiles=2\ntb_logdir="tb_logs"\n'
            before = Path.cwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {"AGENTD_REINVENT_PATH": str(root)}), patch.object(generation, "get_chat_model", return_value=object()), patch.object(generation, "LLMChain", return_value=SimpleNamespace(run=lambda **kwargs: generated)):
                    output = generation.update_reinvent_config.invoke({"model_type": "Mol2Mol"})
                content = tomllib.loads(Path(output).read_text())
                self.assertNotIn("tb_logdir", content["parameters"])
                self.assertIn('# tb_logdir="tb_logs"', Path(output).read_text())
                self.assertEqual(content["parameters"]["num_smiles"], 2)
            finally:
                os.chdir(before)

    def test_pooling_uses_official_text_deduplication(self):
        import pandas as pd
        import mcp_agent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pd.DataFrame({"SMILES": ["CCO"]}).to_csv(root / "Reinvent_sampling.csv", index=False)
            pd.DataFrame({"SMILES": ["OCC", "CCO"]}).to_csv(root / "Mol2Mol_sampling.csv", index=False)
            output = mcp_agent._combine_pool_csvs(root, root / "combined.csv")
            frame = pd.read_csv(output)
            self.assertEqual(len(frame), 2)
            self.assertEqual(frame.SMILES.tolist(), ["CCO", "OCC"])

    def test_pooling_keeps_official_arguments_except_windows_stdin(self):
        import mcp_agent
        import agentD.tools.generation as gen
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "configs/toml").mkdir(parents=True)
            (root / "configs/toml/sampling.toml").write_text('run_type="sampling"')
            cfg = root / "sample.toml"
            cfg.write_text("num_smiles = 2")
            before = Path.cwd()
            try:
                with patch.dict(os.environ, {"AGENTD_REINVENT_PATH": str(root)}), patch.object(gen, "save_smi_for_mol2mol"), patch.object(gen, "update_reinvent_config", return_value=str(cfg)), patch("mcp_agent.subprocess.run") as run:
                    mcp_agent._run_reinvent_pooling(run_dir=root, initial_smiles="CCO")
                for call in run.call_args_list:
                    expected = {"check": True, "cwd": str(root)}
                    if sys.platform == "win32":
                        expected["stdin"] = subprocess.DEVNULL
                    self.assertEqual(call.kwargs, expected)
            finally:
                os.chdir(before)

    def test_runner_disables_user_site_in_mcp_child(self):
        import asyncio
        import run_agentd
        captured = {}
        def transport(**kwargs):
            captured.update(kwargs)
            raise RuntimeError("captured transport")
        with patch.object(run_agentd, "StdioTransport", transport):
            with self.assertRaisesRegex(RuntimeError, "captured transport"):
                asyncio.run(run_agentd.run_pipeline_mode(run_agentd.DEFAULT_CONFIG.copy()))
        self.assertEqual(captured["env"]["PYTHONNOUSERSITE"], "1")

    def test_task_without_explicit_model_honors_runtime_model(self):
        from run_agentd import load_config

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "task.yaml"
            path.write_text("protein: BCL-2\n")
            with patch.dict(os.environ, {"AGENTD_MODEL": "deepseek-v4-pro"}):
                self.assertEqual(load_config(str(path))["model"], "deepseek-v4-pro")

    def test_direct_pooling_model_is_inherited_by_nested_generation(self):
        import mcp_agent

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            before = Path.cwd()
            try:
                with patch.dict(
                    os.environ,
                    {
                        "AGENTD_MODEL": "deepseek-flash",
                        "AGENTD_REINVENT_PATH": str(root),
                    },
                ):
                    mcp_agent._run_reinvent_pooling(
                        run_dir=root, initial_smiles="CCO", model="deepseek-v4-pro"
                    )
                    self.assertEqual(os.environ["AGENTD_MODEL"], "deepseek-v4-pro")
            finally:
                os.chdir(before)

    def test_official_pool_merge_leaves_validation_to_downstream(self):
        import mcp_agent

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "Reinvent_sampling.csv").write_text("SMILES\nnot-a-molecule\n")
            (root / "Mol2Mol_sampling.csv").write_text("SMILES\nalso-invalid\n")
            output = mcp_agent._combine_pool_csvs(root, root / "combined.csv")
            self.assertTrue(output.is_file())

    def test_runner_reports_failed_tool_result_as_failure(self):
        import asyncio, contextlib, io, json
        from types import SimpleNamespace
        import run_agentd

        class Client:
            def __init__(self, *a, **k):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def list_tools(self):
                return []

            async def call_tool(self, *a, **k):
                return SimpleNamespace(
                    content=[
                        SimpleNamespace(
                            text=json.dumps({"ok": False, "error": "test failure"})
                        )
                    ],
                    is_error=False,
                )

        with (
            patch.object(run_agentd, "Client", Client),
            contextlib.redirect_stdout(io.StringIO()) as out,
        ):
            ok = asyncio.run(
                run_agentd.run_pipeline_mode(run_agentd.DEFAULT_CONFIG.copy())
            )
        self.assertFalse(ok)
        self.assertIn("PIPELINE FAILED", out.getvalue())
        self.assertNotIn("PIPELINE COMPLETED SUCCESSFULLY", out.getvalue())

    def test_reinvent_command_keeps_paths_as_separate_arguments(self):
        from agentD.runtime_config import reinvent_command

        with patch.dict(
            os.environ,
            {"AGENTD_REINVENT_EXECUTABLE": "D:/folder with spaces/reinvent.exe"},
            clear=False,
        ):
            cmd = reinvent_command("config with spaces.toml", "output.log")
            self.assertEqual(cmd[-3:], ["-l", "output.log", "config with spaces.toml"])
            self.assertIn("folder with spaces", cmd[0])

    def test_missing_requested_config_does_not_fall_back_to_default_live_task(self):
        from run_agentd import load_config

        with self.assertRaises(FileNotFoundError):
            load_config("nonexistent-config-654321.yaml")

    def test_official_pooling_returns_command_failure_details(self):
        import mcp_agent
        import agentD.tools.generation as gen

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "configs/toml").mkdir(parents=True)
            (root / "configs/toml/sampling.toml").write_text('run_type="sampling"')
            cfg = root / "sample.toml"
            cfg.write_text("num_smiles = 2")
            cwd = Path.cwd()
            try:
                with (
                    patch.dict(os.environ, {"AGENTD_REINVENT_PATH": str(root)}),
                    patch.object(gen, "save_smi_for_mol2mol"),
                    patch.object(gen, "update_reinvent_config", return_value=str(cfg)),
                    patch(
                        "mcp_agent.subprocess.run",
                        side_effect=subprocess.CalledProcessError(1, ["reinvent"]),
                    ),
                ):
                    result = mcp_agent._run_reinvent_pooling(
                        run_dir=root, initial_smiles="CCO"
                    )
                self.assertTrue(result["ok"])
                self.assertIn("Error occurred", result["results"]["Mol2Mol"])
                self.assertIn("Error occurred", result["results"]["Reinvent"])
            finally:
                os.chdir(cwd)
