from types import SimpleNamespace

from src.models.quantization import select_quantized_engine


def test_prefers_fbgemm():
    backend = SimpleNamespace(supported_engines=['qnnpack', 'fbgemm'], engine='none')
    select_quantized_engine(backend)
    assert backend.engine == 'fbgemm'


def test_falls_back_to_qnnpack():
    backend = SimpleNamespace(supported_engines=['qnnpack'], engine='none')
    select_quantized_engine(backend)
    assert backend.engine == 'qnnpack'


def test_unsupported_engine_has_clear_error():
    backend = SimpleNamespace(supported_engines=['none'], engine='none')
    try:
        select_quantized_engine(backend)
    except RuntimeError as error:
        assert 'fbgemm' in str(error) and 'qnnpack' in str(error)
    else:
        raise AssertionError('Expected an unsupported engine error')
