# Reproduce the focused client suite (Python standard library only).
cd /workspace/template/astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online
export PYTHONDONTWRITEBYTECODE=1
export DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY=/workspace/template/die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so
export DIE_WORKSPACE_ENVELOPE_OLD_LIBRARY=/workspace/template/work/five-os-20261014/source-identity/lru-pre-v2-binaries/libcoarse_die_cabi.so
python3 -m unittest -v test_die_workspace_envelope > /workspace/template/work/lru-python-envelope-20261014/python-unittest.log 2>&1

# Optional pytest command was also tried; pytest is not installed in this image.
python3 -m pytest -q test_die_workspace_envelope.py > /workspace/template/work/lru-python-envelope-20261014/pytest.log 2>&1

# Reproduce the real attention/FFN envelope versus selected-Plan observations.
cd /workspace/template
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=/workspace/template/astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online
export DIE_COARSE_CABI_PATH=/workspace/template/die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so
export DIE_WORKSPACE_ENVELOPE_TEST_LIBRARY=/workspace/template/die/work/lru-workspace-envelope-20261014/build/lib/libcoarse_die_cabi.so
export DIE_WORKSPACE_ENVELOPE_OLD_LIBRARY=/workspace/template/work/five-os-20261014/source-identity/lru-pre-v2-binaries/libcoarse_die_cabi.so
python3 /workspace/template/work/lru-python-envelope-20261014/observe_real_cabi.py > /workspace/template/work/lru-python-envelope-20261014/real-cabi-observed.log 2>&1
