"""Per-model train/predict definitions: one file per entry in
`framework.MODEL_ORDER` (neural_net, xgboost_model, lightgbm_model,
dynamic_ridge, naive_baselines).

IMPORTANT -- do NOT add eager submodule imports here (e.g. no
`from . import neural_net, xgboost_model`). `neural_net.py` imports torch;
`xgboost_model.py`/`lightgbm_model.py` import xgboost/lightgbm;
`dynamic_ridge.py` uses scikit-learn inside the same native-model worker. On macOS,
loading both runtimes in the same process segfaults as soon as either runs
its native code (each bundles its own copy of the LLVM OpenMP runtime).
`pipeline.py` (torch) and `tree_worker.py` (xgboost/lightgbm, run as a
subprocess -- see its docstring) rely on being able to import only the
specific model submodule they need, e.g. `from models.neural_net import ...`
or `from models.xgboost_model import ...`, without pulling in the other
runtime as a side effect of importing this package. Keep this file empty
of submodule imports so that stays true.
"""
