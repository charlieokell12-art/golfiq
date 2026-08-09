"""Safe GolfIQ Kaggle entrypoint.

The original bootstrap could silently fall back to synthetic-only imagery and then
spend many GPU hours training. That is no longer allowed. This wrapper delegates
to production_candidate_bootstrap.py, which audits real labelled inputs and fails
before training when the dataset is not suitable.
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / 'production_candidate_bootstrap.py'), run_name='__main__')
