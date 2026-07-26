from functools import cache

import gguf
import pytest

from tests.model_test_cases import ModelTestCase


@cache
def model_reader(model: ModelTestCase) -> gguf.GGUFReader:
    path = model.model_path
    if not path.is_file():
        pytest.skip(f"{model.family} GGUF model is unavailable: {path}")
    return gguf.GGUFReader(path)
