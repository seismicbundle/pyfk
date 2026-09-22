import importlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from pyfk import Config, SeisModel, SourceModel, calculate_gf
from pyfk.utils.error_message import PyfkError
from pyfk.utils.options import frequency_bin_ceiling

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
import upstream_cases  # noqa: E402


def configuration(**options):
    source_type = options.pop('source_type', 'dc')
    depth = options.pop('depth', 8.)
    model = np.array([[10., 3.4, 5.8, 2.6, 600., 1200.],
                      [0., 3.8, 6.5, 2.8, 600., 1200.]])
    base = dict(model=SeisModel(model=model), source=SourceModel(sdep=depth, srcType=source_type),
                receiver_distance=[5., 15.], npt=256, dt=.05)
    base.update(options)
    return Config(**base)


def array(config):
    return np.array([[tr.data for tr in stream] for stream in calculate_gf(config)])


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf'), -1, True, '2'])
def test_invalid_explicit_ceiling(value):
    with pytest.raises(PyfkError, match='fmax'):
        configuration(fmax=value)


@pytest.mark.parametrize('value', ['nan', 'inf', 'nonsense'])
def test_invalid_environment_ceiling(monkeypatch, value):
    monkeypatch.setenv('PYFK_FMAX_HZ', value)
    with pytest.raises(PyfkError):
        configuration()
    assert configuration(fmax=0).fmax == 0


def test_per_calculation_settings_override_environment(monkeypatch):
    monkeypatch.setenv('PYFK_FMAX_HZ', '2')
    monkeypatch.setenv('PYFK_MIXED_FP32', '1')
    cut = configuration()
    full = configuration(fmax=0, mixed_precision=False)
    monkeypatch.setenv('PYFK_FMAX_HZ', '4')
    assert cut.fmax == 2 and cut.mixed_precision is True
    assert full.fmax == 0 and full.mixed_precision is False
    assert 'fmax=2.0' in repr(cut)
    assert os.environ['PYFK_FMAX_HZ'] == '4'
    a, b = array(full), array(cut)
    assert a.shape == b.shape and not np.array_equal(a, b)


@pytest.mark.parametrize('npt,dt', [(256, .05), (1024, .01), (8192, .01)])
def test_ceiling_bin_boundaries(npt, dt):
    dw = 2 * np.pi / (npt * dt)
    df = dw / (2 * np.pi)
    for index in [1, 3, 17, npt // 2 - 2]:
        hz = index * df
        assert frequency_bin_ceiling(npt // 2, dw, hz) == index + 1
        assert frequency_bin_ceiling(npt // 2, dw, np.nextafter(hz, 0)) == index
    assert frequency_bin_ceiling(npt // 2, dw, 1e308) == npt // 2


@pytest.mark.parametrize('pmin', [0., .1])
@pytest.mark.parametrize('source_type', ['dc', 'ep', 'sf'])
def test_threads_and_fullband_equivalence(monkeypatch, pmin, source_type):
    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    full = array(configuration(pmin=pmin, source_type=source_type, fmax=0))
    monkeypatch.setenv('OMP_NUM_THREADS', '2')
    threaded = array(configuration(pmin=pmin, source_type=source_type, fmax=1e308))
    assert np.isfinite(full).all() and np.isfinite(threaded).all()
    assert full.tobytes() == threaded.tobytes()


def check_cutoff_against_masked_spectrum(monkeypatch, pmin, ceiling):
    gf_module = importlib.import_module('pyfk.gf.gf')
    integration = gf_module.waveform_integration
    captured = {}

    def recording(*args, **kwargs):
        spectrum = integration(*args, **kwargs)
        captured['spectrum'] = spectrum.copy()
        return spectrum

    monkeypatch.setattr(gf_module, 'waveform_integration', recording)
    full_config = configuration(pmin=pmin, fmax=0)
    full = array(full_config)
    spectrum = captured['spectrum']
    frequencies = np.arange(spectrum.shape[-1]) / (full_config.npt * full_config.dt)
    masked = spectrum.copy()
    masked[..., frequencies > ceiling] = 0
    monkeypatch.setattr(gf_module, 'waveform_integration', lambda *args, **kwargs: masked.copy())
    expected = array(configuration(pmin=pmin, fmax=0))
    monkeypatch.setattr(gf_module, 'waveform_integration', integration)
    actual = array(configuration(pmin=pmin, fmax=ceiling))
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == full.shape


@pytest.mark.parametrize('pmin', [0., .1])
@pytest.mark.parametrize('ceiling', [.01, 2.0])
def test_cutoff_matches_independently_masked_full_spectrum(monkeypatch, pmin, ceiling):
    check_cutoff_against_masked_spectrum(monkeypatch, pmin, ceiling)


def test_no_bins_and_timing_metadata():
    full = calculate_gf(configuration(fmax=0))
    cut = calculate_gf(configuration(fmax=2))
    for a, b in zip(full, cut):
        for ta, tb in zip(a, b):
            assert ta.stats == tb.stats
    assert not array(configuration(fmax=.01, filter=(1., 2.))).any()


@pytest.mark.parametrize('source_type', ['dc', 'sf', 'ep'])
def test_directional_components_sum_to_full_wavefield(source_type):
    full = array(configuration(source_type=source_type))
    upward = array(configuration(source_type=source_type, updn='up'))
    downward = array(configuration(source_type=source_type, updn='down'))
    assert np.isfinite(upward).all() and np.isfinite(downward).all()
    assert np.max(abs(upward)) > 0 and np.max(abs(downward)) > 0
    np.testing.assert_allclose(upward + downward, full, rtol=1e-8, atol=np.max(abs(full)) * 1e-10)


def check_reference():
    actual = upstream_cases.collect()
    with np.load(Path(__file__).parent / 'data/upstream-reference.npz') as reference:
        assert set(actual) == set(reference.files)
        for name, value in actual.items():
            assert np.isfinite(value).all()
            expected = reference[name]
            scale = np.max(abs(expected))
            # Upstream itself differs across MSVC/x64 and clang/ARM for the
            # shallow-source case (peak-normalized synthetic error 1.03e-8).
            # Keep headers strict; allow a bounded waveform difference here.
            tolerance = 1e-10 if name.endswith('.headers') else 5e-8
            np.testing.assert_allclose(value, expected, rtol=1e-8,
                atol=max(scale, 1e-30) * tolerance, err_msg=name)


def check_native_reference(path):
    """Strict comparison to unoptimized upstream built on the same platform."""
    actual = upstream_cases.collect()
    with np.load(path) as reference:
        assert set(actual) == set(reference.files)
        for name, value in actual.items():
            expected = reference[name]
            assert np.isfinite(value).all() and np.isfinite(expected).all()
            np.testing.assert_allclose(value, expected, rtol=1e-8,
                atol=max(np.max(abs(expected)), 1e-30) * 1e-10, err_msg=name)


def test_reference_waveforms_and_metadata():
    check_reference()


def test_native_upstream_waveforms_and_metadata():
    path = os.getenv('PYFK_TEST_UPSTREAM_REFERENCE')
    if not path:
        pytest.skip('set PYFK_TEST_UPSTREAM_REFERENCE to a native upstream NPZ')
    check_native_reference(path)


def test_upstream_api_signatures():
    import pyfk
    record = json.loads((Path(__file__).parent / 'data/upstream-reference.json').read_text())
    assert pyfk.__all__ == record['exports']
    for name, expected in record['parameters'].items():
        actual = inspect.signature(getattr(pyfk, name)).parameters
        for parameter in expected:
            current = actual[parameter['name']]
            assert current.kind.name == parameter['kind']
            assert repr(current.default) == parameter['default']
        assert list(actual)[:len(expected)] == [p['name'] for p in expected]


@pytest.mark.parametrize('value', ['0', '-1', 'garbage'])
def test_bad_cuda_chunk_count_fails_before_loading_cuda(monkeypatch, value):
    monkeypatch.setenv('CUDA_DIVIDE_NUM', value)
    with pytest.raises(PyfkError, match='CUDA_DIVIDE_NUM'):
        calculate_gf(configuration(cuda=True))


@pytest.mark.cuda
@pytest.mark.parametrize('pmin,source_type,ceiling,chunks,mixed', [
    (0., 'dc', 0., 1, False), (0., 'dc', 2., 7, False),
    (.1, 'sf', 2., 3, False), (0., 'ep', .01, 1, False),
    (0., 'dc', 2., 7, True), (0., 'dc', .01, 1, True),
])
def test_cuda_cpu_agreement(monkeypatch, pmin, source_type, ceiling, chunks, mixed):
    monkeypatch.setenv('CUDA_DIVIDE_NUM', str(chunks))
    config = dict(pmin=pmin, source_type=source_type, fmax=ceiling)
    cpu = array(configuration(**config))
    gpu = array(configuration(**config, cuda=True, mixed_precision=mixed))
    assert np.isfinite(gpu).all()
    relative = np.max(abs(cpu - gpu)) / np.max(abs(cpu))
    assert relative < (1e-3 if mixed else 1e-8)


@pytest.mark.cuda
def test_cuda_chunk_invariance_and_empty_band(monkeypatch):
    for mixed in [False, True]:
        monkeypatch.setenv('CUDA_DIVIDE_NUM', '1')
        first = array(configuration(cuda=True, fmax=2, mixed_precision=mixed))
        monkeypatch.setenv('CUDA_DIVIDE_NUM', '7')
        second = array(configuration(cuda=True, fmax=2, mixed_precision=mixed))
        np.testing.assert_array_equal(first, second)
    assert not array(configuration(cuda=True, fmax=.01, filter=(1., 2.))).any()


@pytest.mark.cuda
def test_cuda_projection_has_no_contracted_fp64_operations():
    from pyfk.gf.cuda.waveform_integration_cuda import get_sum_waveform_bins
    assert all('fma.rn.f64' not in asm for asm in get_sum_waveform_bins.inspect_asm().values())


@pytest.mark.cuda
def test_cuda_matches_upstream_case_matrix():
    actual = upstream_cases.collect(cuda=True)
    with np.load(Path(__file__).parent / 'data/upstream-reference.npz') as reference:
        for name, value in actual.items():
            assert np.isfinite(value).all()
            expected = reference[name]
            np.testing.assert_allclose(value, expected, rtol=1e-7,
                atol=max(np.max(abs(expected)), 1e-30) * 1e-8, err_msg=name)


@pytest.mark.mpi
def test_mpi_reference_and_frequency_ceiling(monkeypatch):
    from pyfk import mpi_info
    assert mpi_info().startswith('MPI installed correctly')
    check_reference()
    if os.getenv('PYFK_TEST_UPSTREAM_REFERENCE'):
        check_native_reference(os.environ['PYFK_TEST_UPSTREAM_REFERENCE'])
    check_cutoff_against_masked_spectrum(monkeypatch, .1, 2.)
