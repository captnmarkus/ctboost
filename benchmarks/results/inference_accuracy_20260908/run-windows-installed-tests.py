import os
import pathlib
import sys

os.environ['CTBOOST_HIST_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import ctboost
import psutil
import pytest

psutil.Process().cpu_affinity([2, 3, 4, 10, 11, 12])
root = pathlib.Path(__file__).parent / 'installed-tests'
os.chdir(root)
sys.path.insert(0, str(root))
print('Installed candidate:', ctboost.__file__, flush=True)
raise SystemExit(pytest.main(['tests', '-q', '--import-mode=importlib']))
