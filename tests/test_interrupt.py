"""Run interruption checks in a child so a native abort cannot kill pytest."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('pmin', [0., .1])
def test_interrupt_is_python_exception(pmin):
    script = '''
import _thread
import os
import signal
import threading
import time
import numpy as np
from pyfk import Config, SeisModel, SourceModel, calculate_gf
model = np.array([[10.,3.4,5.8,2.6,600.,1200.],[0.,3.8,6.5,2.8,600.,1200.]])
config = Config(model=SeisModel(model=model),source=SourceModel(sdep=8.),
                receiver_distance=[5.,15.],npt=16384,dt=.01,pmin=PMIN,fmax=0)
def interrupt():
    if os.name == 'posix':
        os.kill(os.getpid(), signal.SIGINT)
    else:
        _thread.interrupt_main()
timer = threading.Timer(.1, interrupt)
timer.daemon = True
try:
    started = time.monotonic()
    timer.start()
    calculate_gf(config)
except KeyboardInterrupt:
    assert time.monotonic() - started < 5, 'interruption was not responsive'
    print('CLEAN_INTERRUPT')
else:
    raise SystemExit('calculation completed without exercising interruption')
finally:
    timer.cancel()
'''.replace('PMIN', repr(pmin))
    env = dict(os.environ, OMP_NUM_THREADS='2', PYFK_USE_CUDA='0')
    result = subprocess.run([sys.executable, '-c', script], env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'CLEAN_INTERRUPT' in result.stdout
