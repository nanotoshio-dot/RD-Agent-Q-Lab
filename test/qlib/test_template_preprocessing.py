"""Backtests that are compared with each other must prepare their features the same way."""

from pathlib import Path
from types import SimpleNamespace

import jinja2
import jinja2.meta
import numpy as np
import pandas as pd
import pytest
import yaml

import rdagent.scenarios.qlib.developer.utils as developer_utils
from rdagent.components.runner import CachedRunner
from rdagent.scenarios.qlib.developer.utils import qlib_run_cache_key, to_feature_frame

pytestmark = pytest.mark.offline

EXPERIMENT_DIR = Path(__file__).resolve().parents[2] / "rdagent" / "scenarios" / "qlib" / "experiment"
TEMPLATES = sorted(EXPERIMENT_DIR.glob("*_template/conf_*.yaml"))
ENV = {
    "train_start": "2011-02-01",
    "train_end": "2016-03-31",
    "valid_start": "2016-04-01",
    "valid_end": "2017-12-31",
    "test_start": "2018-01-01",
    "test_end": "2020-06-30",
    "feature_names": "['A']",
    "feature_expressions": "['$close']",
    "num_features": "21",
    "dataset_cls": "DatasetH",
}


def parsed(name: str, env: dict[str, str] = ENV) -> dict:
    # Same rule as qlib.cli.run.render_template: only variables present in the environment reach Jinja.
    [path] = [path for path in TEMPLATES if path.name == name]
    template = jinja2.Template(path.read_text())
    variables = jinja2.meta.find_undeclared_variables(template.environment.parse(path.read_text()))
    return yaml.safe_load(template.render({key: env[key] for key in variables if key in env}))


def handler(config: dict) -> dict:
    return config["task"]["dataset"]["kwargs"]["handler"]["kwargs"]


@pytest.mark.parametrize("env", [{}, ENV], ids=["defaults", "market dates"])
def test_all_templates_share_the_preprocessing(env):
    handlers = {path.name: handler(parsed(path.name, env)) for path in TEMPLATES}
    expected = handlers["conf_baseline.yaml"]

    assert [step["class"] for step in expected["infer_processors"]] == ["RobustZScoreNorm", "Fillna"]
    for name, kwargs in handlers.items():
        for key in ("infer_processors", "learn_processors", "start_time", "end_time", "instruments"):
            assert kwargs[key] == expected[key], (name, key)


def test_the_normalization_is_fitted_on_the_training_period():
    normalize = handler(parsed("conf_combined_factors.yaml"))["infer_processors"][0]["kwargs"]

    assert str(normalize["fit_start_time"]) == ENV["train_start"]
    assert str(normalize["fit_end_time"]) == ENV["train_end"]


def test_the_factor_backtest_differs_from_the_baseline_only_by_the_new_factors():
    combined, baseline = parsed("conf_combined_factors.yaml"), parsed("conf_baseline.yaml")
    loaders = handler(combined)["data_loader"]["kwargs"]["dataloader_l"]
    static = [loader for loader in loaders if loader["class"].endswith("StaticDataLoader")]
    assert len(static) == 1
    loaders.remove(static[0])

    assert combined == baseline


def test_factor_columns_become_float_features_and_keep_partial_rows():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2020-01-03"), "B"), (pd.Timestamp("2020-01-02"), "A")], names=["datetime", "instrument"]
    )
    factors = pd.DataFrame(
        {
            "flag": pd.Series([True, False], index=index),
            "nullable": pd.Series([1.5, None], index=index, dtype="Float64"),
            "words": pd.Series(["2", "not a number"], index=index, dtype=object),
        }
    )
    factors = pd.concat([factors, pd.DataFrame({"flag": [3.0, np.nan]}, index=index)], axis=1)

    frame = to_feature_frame(factors)

    assert list(frame.columns) == [("feature", "nullable"), ("feature", "words"), ("feature", "flag")]
    assert set(frame.dtypes.astype(str)) == {"float64"}
    assert frame.index.is_monotonic_increasing
    assert len(frame) == 2  # a row one factor lacks keeps the others
    assert frame.loc[(pd.Timestamp("2020-01-03"), "B"), ("feature", "flag")] == 3.0  # the last "flag" wins
    assert frame.loc[(pd.Timestamp("2020-01-03"), "B"), ("feature", "words")] == 2.0
    assert np.isnan(frame.loc[(pd.Timestamp("2020-01-02"), "A"), ("feature", "words")])


def test_missing_dates_stay_missing_and_a_column_without_numbers_is_named(monkeypatch):
    warnings = []
    monkeypatch.setattr(developer_utils.logger, "warning", warnings.append)
    index = pd.MultiIndex.from_product([[pd.Timestamp("2020-01-02")], ["A", "B"]], names=["datetime", "instrument"])
    factors = pd.DataFrame({"age": pd.to_timedelta(["1d", None]), "label": ["up", "down"]}, index=index)

    frame = to_feature_frame(factors)

    assert np.isnan(frame[("feature", "age")].iloc[1])
    assert frame[("feature", "label")].isna().all()
    assert len(warnings) == 1
    assert "label" in warnings[0]
    assert "age" not in warnings[0]


def experiment(files: dict[str, str]) -> SimpleNamespace:
    task = SimpleNamespace(get_task_information=lambda: "momentum")
    return SimpleNamespace(
        based_experiments=[], sub_tasks=[task], experiment_workspace=SimpleNamespace(file_dict=files)
    )


class Runner(CachedRunner):
    def develop(self, exp):
        return exp


def test_a_changed_template_gets_a_new_cache_key():
    runner = Runner(scen=None)
    first = experiment({"conf_combined_factors.yaml": "old", "model.py": "a"})

    key = qlib_run_cache_key(runner, first)

    assert key == qlib_run_cache_key(runner, experiment({"conf_combined_factors.yaml": "old", "model.py": "b"}))
    assert key != qlib_run_cache_key(runner, experiment({"conf_combined_factors.yaml": "new", "model.py": "a"}))
    assert key != runner.get_cache_key(first)


def test_qlib_normalization_accepts_what_factor_code_may_write():
    processor = pytest.importorskip("qlib.data.dataset.processor")
    days = pd.date_range("2020-01-01", periods=6)
    index = pd.MultiIndex.from_product([days, ["A", "B"]], names=["datetime", "instrument"])
    factors = pd.DataFrame(
        {
            "plain": np.arange(12.0),
            "infinite": [np.inf, -np.inf] + [1.0, 2.0] * 5,
            "constant": np.ones(12),
            "empty": np.full(12, np.nan),
            "flag": [True, False] * 6,
        },
        index=index,
    )
    frame = to_feature_frame(factors)
    normalize = processor.RobustZScoreNorm(
        fit_start_time=days[0], fit_end_time=days[-1], fields_group="feature", clip_outlier=True
    )
    normalize.fit(frame)

    prepared = processor.Fillna(fields_group="feature")(normalize(frame))

    assert np.isfinite(prepared.to_numpy()).all()
    assert prepared.abs().to_numpy().max() <= 3
