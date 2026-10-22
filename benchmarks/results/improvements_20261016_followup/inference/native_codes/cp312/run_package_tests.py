"""Run isolated package tests with the repository benchmark helpers visible."""
from pathlib import Path
import runpy
import sys

repository = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repository))
runpy.run_path(str(repository / '.tmp/improvements-20261016/run_tests.py'), run_name='__main__')
