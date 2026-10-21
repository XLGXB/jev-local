"""Offline constructor configuration contract; no native checkpoint execution."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import AutoTokenizer
import safetensors.torch

from decision_ft.backends import LayaBackend


class TinyNativeShape(nn.Module):
    def __init__(self, expose_compile):
        super().__init__()
        self.encoder = nn.Linear(1, 1)
        self.encoder.config = SimpleNamespace(**({'reference_compile': True} if expose_compile else {}))
        self.act_head = nn.Linear(1, 1)
        self.register_buffer('temperature', torch.tensor([2.0, 2.0, 2.0]))


class NativeConfigTests(unittest.TestCase):
    def test_laya_uses_eager_path_when_available_without_adding_unknown_config(self):
        for expose_compile in (True, False):
            with self.subTest(expose_compile=expose_compile), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / 'encoder').mkdir()
                (root / 'encoder/config.json').write_text('{}', encoding='utf-8')
                (root / 'rl_agent_config.json').write_text(json.dumps({'encoder': 'fixture-only'}), encoding='utf-8')
                (root / 'model.safetensors').write_bytes(b'not-a-real-checkpoint')
                model = TinyNativeShape(expose_compile)
                common = ModuleType('laya.common')
                common.__file__ = __file__
                common.build_model = lambda *args, **kwargs: model
                package = ModuleType('laya')
                package.common = common
                with patch.dict(sys.modules, {'laya': package, 'laya.common': common}), \
                     patch.object(AutoTokenizer, 'from_pretrained', return_value=object()) as tokenizer, \
                     patch('safetensors.torch.load_file', return_value=model.state_dict()), \
                     patch.object(model, 'load_state_dict', wraps=model.load_state_dict) as loader:
                    backend = LayaBackend(str(root), 'cpu')
                loader.assert_called_once()
                self.assertTrue(loader.call_args.kwargs['strict'])
                self.assertTrue(tokenizer.call_args.kwargs['local_files_only'])
                self.assertFalse(tokenizer.call_args.kwargs['trust_remote_code'])
                self.assertEqual(backend.provenance['reference_compile'], False if expose_compile else None)
                self.assertEqual(hasattr(model.encoder.config, 'reference_compile'), expose_compile)
                self.assertTrue(all(not p.requires_grad for p in model.act_head.parameters()))
                self.assertTrue(torch.equal(model.temperature, torch.ones(3)))


if __name__ == '__main__':
    unittest.main()
