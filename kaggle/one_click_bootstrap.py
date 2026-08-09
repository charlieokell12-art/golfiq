"""Safe one-click GolfIQ Kaggle entrypoint.

This runs the full production-candidate pipeline: real-data audit, legal public
train-only augmentation, sanity training, full training, held-out test metrics,
and ONNX export. It deliberately stops before training if real labelled inputs
are not good enough.
"""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE / 'full_production_candidate.py'), run_name='__main__')
