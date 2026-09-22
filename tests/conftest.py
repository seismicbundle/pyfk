import os
import pytest


@pytest.fixture(autouse=True)
def deterministic_options(monkeypatch):
    for key in ['PYFK_FMAX_HZ', 'PYFK_MIXED_FP32', 'CUDA_DIVIDE_NUM']:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv('PYFK_USE_CUDA', '0')
    monkeypatch.setenv('OMP_NUM_THREADS', '2')


def pytest_collection_modifyitems(config, items):
    if os.getenv('PYFK_TEST_CUDA') != '1':
        for item in items:
            if 'cuda' in item.keywords:
                item.add_marker(pytest.mark.skip(reason='set PYFK_TEST_CUDA=1 for GPU validation'))
