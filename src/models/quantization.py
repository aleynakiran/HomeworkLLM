"""Select a supported CPU engine before creating or loading quantized layers."""


def select_quantized_engine(backend):
    # Prefer the x86 engine, then use QNNPACK (including on Apple Silicon).
    for engine in ('fbgemm', 'qnnpack'):
        if engine in backend.supported_engines:
            backend.engine = engine
            return
    raise RuntimeError('INT8 quantization requires a PyTorch build with fbgemm or qnnpack support.')
