"""Run a fixed set of public-API cases and save the outputs.

Usage: python validation/upstream_cases.py OUTPUT.npz
Run with unoptimized upstream installed to create a reference archive.
"""
import inspect
import json
from pathlib import Path
import sys
import numpy as np
import pyfk

CASES = [
    {}, {'pmin': .1}, {'src_type': 'ep'}, {'src_type': 'sf'},
    {'depth': .25}, {'depth': 19.9}, {'rdep': 12.}, {'smth': 2.},
    {'filter': (.3, 1.)}, {'suppression_sigma': 3.},
    {'degrees': True, 'planet_radius': 3390., 'receiver_distance': [.1, .3]},
    {'updn': 'up'}, {'updn': 'down'},
]


def collect(cuda=False):
    payload = {}
    for index, original in enumerate(CASES):
        options = dict(original)
        depth = options.pop('depth', 8.)
        src_type = options.pop('src_type', 'dc')
        source = pyfk.SourceModel(sdep=depth, srcType=src_type)
        model = pyfk.SeisModel(model=np.array([[10., 3.4, 5.8, 2.6, 600., 1200.],
                                             [0., 3.8, 6.5, 2.8, 600., 1200.]]))
        values = dict(model=model, source=source, receiver_distance=[5., 15.], npt=256, dt=.05)
        values.update(options)
        config = pyfk.Config(**values, cuda=cuda)
        gf = pyfk.calculate_gf(config)
        payload[f'case{index}.gf'] = np.array([[tr.data for tr in st] for st in gf])
        payload[f'case{index}.headers'] = np.array([[st[0].stats.delta] +
            [st[0].stats.sac[k] for k in ['b', 'e', 'dist', 't1', 't2', 'user1', 'user2']] for st in gf])
        if src_type == 'dc':
            source.update_source_mechanism([2., 30., 60., 90.])
            stf = pyfk.generate_source_time_function(.2, .5, gf[0][0].stats.delta)
            syn = pyfk.calculate_sync(gf, config, 30., stf)
            payload[f'case{index}.sync'] = np.array([[tr.data for tr in st] for st in syn])
    return payload


if __name__ == '__main__':
    destination = Path(sys.argv[1])
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **collect())
    parameters = {}
    for name in pyfk.__all__:
        try:
            parameters[name] = [dict(name=p.name, kind=p.kind.name, default=repr(p.default))
                                for p in inspect.signature(getattr(pyfk, name)).parameters.values()]
        except ValueError:
            pass
    destination.with_suffix('.json').write_text(json.dumps(dict(exports=pyfk.__all__, parameters=parameters,
        version=pyfk.__version__, upstream_commit='810e77dee6e7fe12ebac41832dfc17e6b7bab29f'), indent=2) + '\n', encoding='utf-8', newline='\n')
