"""Prepare an unoptimized upstream checkout for a native reference build.

Usage: python validation/prepare_upstream.py UPSTREAM_GIT_CHECKOUT EMPTY_DESTINATION
Then install the destination in a separate environment and run upstream_cases.py.
"""
import argparse
import io
from pathlib import Path
import subprocess
import zipfile

REVISION = '810e77dee6e7fe12ebac41832dfc17e6b7bab29f'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream_checkout', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        parser.error('destination must be empty; existing files are never overwritten')
    destination.mkdir(parents=True, exist_ok=True)
    blob = subprocess.check_output(['git', '-C', str(args.upstream_checkout),
                                    'archive', '--format=zip', REVISION])
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for name in archive.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError('archive path escapes destination')
        archive.extractall(destination)
    release = Path(__file__).resolve().parents[1]
    for name in ['pyfk/utils/complex.pxd', 'setup.py', 'build_backend.py',
                 'pyproject.toml', 'NOTICE', 'MANIFEST.in']:
        (destination / name).write_bytes((release / name).read_bytes())
    for name in ['pyfk/gf/waveform_integration.pyx', 'pyfk/taup/taup.pyx']:
        path = destination / name
        text = path.read_text().replace('cdivision=True', 'cdivision=True, cpow=True')
        text = text.replace('temppp.T[jmat, kmat]', 'temppp[kmat, jmat]')
        path.write_text(text)
    path = destination / 'pyproject.toml'
    text = path.read_text().replace('0.3.1+seismicbundle.1', '0.3.1+reference')
    text = text.replace('"Cython>=3.0,<3.4"]', '"Cython>=3.0,<3.4", "cysignals>=1.12"]')
    text = text.replace('dependencies = [', 'dependencies = ["cysignals>=1.12", ', 1)
    path.write_text(text)
    path = destination / 'setup.py'
    path.write_text(path.read_text().replace('include_dirs = [np.get_include()]',
        'import cysignals\ninclude_dirs = [np.get_include(), str(Path(cysignals.__file__).parent / "include")]'))
    print(f'Unoptimized reference prepared from {REVISION} at {destination}')


if __name__ == '__main__':
    main()
