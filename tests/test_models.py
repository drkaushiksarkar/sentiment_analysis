from sentiment_package.imdb import models as imdb_models
from sentiment_package.sarcasm import models as sarcasm_models


def test_imdb_dense_model_input_shape() -> None:
    cfg = imdb_models.DenseModelConfig(vocab_size=500, max_length=32)
    model = imdb_models.build_dense_model(cfg)
    model.build((None, cfg.max_length))
    assert model.input_shape == (None, cfg.max_length)
    assert model.output_shape == (None, 1)


def test_sarcasm_conv_model_layers() -> None:
    cfg = sarcasm_models.ConvSarcasmConfig(vocab_size=1000, max_length=16)
    model = sarcasm_models.build_conv_model(cfg)
    model.build((None, cfg.max_length))
    layer_types = [layer.__class__.__name__ for layer in model.layers]
    assert "Embedding" in layer_types[0]
    assert "Conv1D" in layer_types[2]
    assert model.output_shape == (None, 1)
