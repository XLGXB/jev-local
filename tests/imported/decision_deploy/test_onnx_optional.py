"""Only an operator/export plumbing smoke when optional packages exist; not ModernBERT."""
import pytest

def test_optional_onnx_roundtrip(tmp_path):
    onnx=pytest.importorskip('onnx',reason='ONNX package unavailable in validation environment')
    ort=pytest.importorskip('onnxruntime',reason='ONNX Runtime unavailable in validation environment')
    pytest.importorskip('onnxscript',reason='dynamo exporter dependency unavailable')
    import torch,numpy as np
    class Toy(torch.nn.Module):
        def forward(self,x):return torch.tanh(x)*2
    m=Toy().eval();x=torch.randn(1,8);path=tmp_path/'toy.onnx'
    torch.onnx.export(m,(x,),str(path),input_names=['x'],output_names=['z'],dynamo=True,opset_version=18)
    onnx.checker.check_model(str(path))
    session=ort.InferenceSession(str(path),providers=['CPUExecutionProvider'])
    np.testing.assert_allclose(session.run(None,{'x':x.numpy()})[0],m(x).numpy(),atol=1e-5)
