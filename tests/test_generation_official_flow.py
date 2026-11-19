"""Official generation flow with only DeepSeek and local-path adapters."""
import builtins
from types import SimpleNamespace
from unittest.mock import Mock
import pytest


@pytest.mark.parametrize('line, expected_line', [
    ('tb_logdir="tb_logs"', '# tb_logdir="tb_logs"'),
    ('  tb_logdir = "tb_logs" # comment', '  # tb_logdir = "tb_logs" # comment'),
    ('#tb_logdir="tb_logs"', '#tb_logdir="tb_logs"'),
])
def test_config_only_comments_sampling_log_field_without_importing_reinvent(
        tmp_path, monkeypatch, line, expected_line):
    from agentD.tools import generation

    monkeypatch.chdir(tmp_path)
    template = tmp_path / 'configs/toml/sampling.toml'
    template.parent.mkdir(parents=True)
    template.write_text('run_type="sampling"', encoding='utf-8')
    generated = ('run_type="sampling"\ndevice="cuda:0"\n[parameters]\n'
                 'model_file="D:/local/model.prior"\nnum_smiles=2\n' + line + '\n')
    monkeypatch.setattr(generation, 'load_runtime', lambda: SimpleNamespace(reinvent_path=tmp_path))
    model = Mock(return_value=object())
    monkeypatch.setattr(generation, 'get_chat_model', model)
    chain = Mock(return_value=generated)
    monkeypatch.setattr(generation, 'LLMChain', lambda **kwargs: SimpleNamespace(run=chain))
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith(('reinvent', 'scipy')), 'Heavy schema import restored accidentally'
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', guarded_import)
    result = generation.update_reinvent_config.invoke({'model_type': 'Mol2Mol'})
    assert (tmp_path / result).read_text() == generated.replace(line, expected_line)
    model.assert_called_once_with(temperature=0.0)
    assert chain.call_args.kwargs['prefix'] == tmp_path.as_posix()
