import os
from pathlib import Path

import pandas as pd
import pytest

import rdagent.scenarios.qlib.experiment.utils as data_utils
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.utils.env import CondaConf, EnvResult, QlibLocalEnv, QTDockerEnv

pytestmark = pytest.mark.offline

TEMPLATE_DIR = Path(data_utils.__file__).parent / "factor_data_template"
OUTPUTS = ("daily_pv_all.h5", "daily_pv_debug.h5")


class FakeEnv:
    def __init__(self, exit_code: int = 0, write_outputs: bool = True, conf: object = None) -> None:
        self.exit_code = exit_code
        self.write_outputs = write_outputs
        self.conf = conf
        self.prepared = False
        self.calls = []

    def prepare(self) -> None:
        self.prepared = True

    def run(self, entry: str | None = None, local_path: str = ".", **kwargs) -> EnvResult:
        build = Path(local_path)
        self.calls.append({"entry": entry, "build": build, "has_script": (build / "generate.py").is_file()})
        if self.write_outputs:
            for name in OUTPUTS:
                (build / name).write_text(f"{name} built in {build.name}")
        return EnvResult("generation log", self.exit_code, 0.1)


@pytest.fixture
def folders(tmp_path, monkeypatch):
    data = tmp_path / "workspace" / "factor_data"
    debug = tmp_path / "workspace" / "factor_data_debug"
    monkeypatch.setattr(data_utils.FACTOR_COSTEER_SETTINGS, "data_folder", str(data))
    monkeypatch.setattr(data_utils.FACTOR_COSTEER_SETTINGS, "data_folder_debug", str(debug))
    return data, debug


def use_env(monkeypatch, env: FakeEnv) -> None:
    monkeypatch.setattr(data_utils, "_factor_data_env", lambda: env)


def test_success_publishes_both_folders_from_a_private_build(folders, monkeypatch):
    env = FakeEnv()
    use_env(monkeypatch, env)

    data_utils.generate_data_folder_from_qlib()

    for folder, output in zip(folders, OUTPUTS, strict=True):
        assert sorted(path.name for path in folder.iterdir()) == ["README.md", "daily_pv.h5"]
        assert (folder / "daily_pv.h5").read_text().startswith(output)
    (call,) = env.calls
    assert env.prepared
    assert call["entry"] == "python generate.py"
    assert call["has_script"]
    assert call["build"].parent == folders[0].parent
    assert not call["build"].exists()


@pytest.mark.parametrize("env", [FakeEnv(exit_code=1), FakeEnv(write_outputs=False)])
def test_failed_generation_publishes_nothing(folders, monkeypatch, env):
    use_env(monkeypatch, env)

    with pytest.raises(RuntimeError, match="generation log"):
        data_utils.generate_data_folder_from_qlib()

    assert not any(folder.exists() for folder in folders)
    assert not any(path.name.startswith(".factor_data_build_") for path in folders[0].parent.iterdir())


def test_generation_never_writes_into_the_template_folder(folders, monkeypatch):
    before = sorted(path.name for path in TEMPLATE_DIR.iterdir())
    use_env(monkeypatch, FakeEnv())

    data_utils.generate_data_folder_from_qlib()

    assert sorted(path.name for path in TEMPLATE_DIR.iterdir()) == before


def test_regeneration_replaces_leftover_folders(folders, monkeypatch):
    data, debug = folders
    data.mkdir(parents=True)
    (data / "stale.h5").write_text("previous run")
    use_env(monkeypatch, FakeEnv())

    data_utils.generate_data_folder_from_qlib()

    assert sorted(path.name for path in data.iterdir()) == ["README.md", "daily_pv.h5"]
    assert debug.is_dir()


def test_conda_path_is_read_again_after_prepare_creates_the_env(folders, monkeypatch):
    conf = CondaConf(conda_env_name="env-that-does-not-exist-yet")
    conf.bin_path = ""
    refreshed = []

    def update_bin_path(self) -> None:
        refreshed.append(True)
        self.bin_path = "/opt/conda/envs/new/bin"

    monkeypatch.setattr(CondaConf, "_update_bin_path", update_bin_path)
    use_env(monkeypatch, FakeEnv(conf=conf))

    data_utils.generate_data_folder_from_qlib()

    assert refreshed == [True]
    assert conf.bin_path == "/opt/conda/envs/new/bin"


def clear_market_env(monkeypatch) -> None:
    for name in [key for key in os.environ if key.startswith("RD_MARKET_")]:
        monkeypatch.delenv(name)


def test_refresh_readme_replaces_a_stale_copy(tmp_path):
    (tmp_path / "README.md").write_text("$factor: factor value of the stock on that day.")

    data_utils.refresh_readme(tmp_path)

    assert (tmp_path / "README.md").read_bytes() == (TEMPLATE_DIR / "README.md").read_bytes()


@pytest.mark.parametrize(("market_id", "refreshed"), [("china", True), (None, False)])
def test_only_market_runs_refresh_the_published_readme(folders, monkeypatch, market_id, refreshed):
    clear_market_env(monkeypatch)
    if market_id:
        monkeypatch.setenv("RD_MARKET_ID", market_id)
    for folder in folders:
        folder.mkdir(parents=True)
        (folder / "README.md").write_text("hand-made description")
    monkeypatch.setattr(data_utils, "get_file_desc", lambda p, variable_list=[]: p.name)

    assert data_utils.get_data_folder_intro() == "README.md"

    template = (TEMPLATE_DIR / "README.md").read_text()
    for folder in folders:
        assert ((folder / "README.md").read_text() == template) is refreshed


def test_template_readme_names_the_columns_and_the_adjustment_factor():
    readme = (TEMPLATE_DIR / "README.md").read_text()

    assert 'df["$close"], never df["close"]' in readme
    assert 'groupby(level="instrument")' in readme
    assert "$factor: price adjustment factor" in readme


def test_the_readme_example_writes_a_result_in_the_expected_format(tmp_path, monkeypatch):
    # The README is part of the coder's prompt, so its example must run and compute per instrument.
    readme = (TEMPLATE_DIR / "README.md").read_text()
    example = readme.split("## A complete factor.py\n```Python\n", 1)[1].split("```", 1)[0]
    days = pd.date_range("2020-01-01", periods=6, name="datetime")
    index = pd.MultiIndex.from_product([days, ["A", "B"]], names=["datetime", "instrument"])
    volume = [1.0, 100.0] * 6
    data = pd.DataFrame({name: 1.0 for name in ("$open", "$close", "$high", "$low", "$factor")}, index=index)
    data["$volume"] = volume
    data.to_hdf(tmp_path / "daily_pv.h5", key="data")
    monkeypatch.chdir(tmp_path)

    exec(example, {})  # noqa: S102 - the README's own example

    result = pd.read_hdf(tmp_path / "result.h5", key="data")
    assert list(result.index.names) == ["datetime", "instrument"]
    assert list(result.columns) == ["Volume_MA5"]
    assert result.xs("B", level="instrument")["Volume_MA5"].dropna().tolist() == [100.0, 100.0]


def test_plain_runs_keep_the_upstream_docker_generation(monkeypatch):
    clear_market_env(monkeypatch)

    assert isinstance(data_utils._factor_data_env(), QTDockerEnv)


def test_market_runs_generate_where_the_models_run(monkeypatch):
    clear_market_env(monkeypatch)
    monkeypatch.setenv("RD_MARKET_ID", "us-sp500")
    monkeypatch.setenv("RD_MARKET_PROVIDER_URI", "/data/us_data")
    monkeypatch.setattr(MODEL_COSTEER_SETTINGS, "env_type", "local")

    assert isinstance(data_utils._factor_data_env(), QlibLocalEnv)


def test_docker_generation_is_refused_for_us_markets(monkeypatch):
    clear_market_env(monkeypatch)
    monkeypatch.setenv("RD_MARKET_ID", "us-sp500")
    monkeypatch.setenv("RD_MARKET_PROVIDER_URI", "/data/us_data")
    monkeypatch.setattr(MODEL_COSTEER_SETTINGS, "env_type", "docker")

    with pytest.raises(ValueError, match="cannot run in Docker"):
        data_utils._factor_data_env()
