# If a the preSample preprocessing pipeline completes
# but now you just want ot manipuatle the quarto code
# you can just use this command to generate and test the
# quarto code

python - <<'PY'
from pipeline_reporting import generate_quarto_report
from pipeline_config import PipelineConfig
from path_config import PathConfig

config = PipelineConfig.load(
    "/home/tonya/test_scanpy_jordan_08032026/test_B2107/"
    "workspace_files/"
    "pipeline_config_testSample_1_independent_analysis_08232026.pkl"
)

paths = PathConfig.from_config(config)

generate_quarto_report(
    paths=paths,
    config=config,
)
PY
