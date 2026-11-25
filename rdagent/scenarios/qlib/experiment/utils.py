import os
import random
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.model_coder.conf import MODEL_COSTEER_SETTINGS
from rdagent.core.market.runtime import (
    current_market_id,
    ensure_docker_supports_market,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import (
    CondaConf,
    Env,
    QlibCondaConf,
    QlibCondaEnv,
    QlibDockerConf,
    QlibLocalConf,
    QlibLocalEnv,
    QTDockerEnv,
)

FACTOR_DATA_TEMPLATE = Path(__file__).parent / "factor_data_template"


def _factor_data_env() -> Env:
    """Pick where generate.py runs.

    Plain RD-Agent runs keep the upstream Docker generation. Market runs use the execution
    environment of the models, because the Docker container would build the default China data.
    """
    market_id = current_market_id()
    if market_id is None:
        return QTDockerEnv()
    env_type = MODEL_COSTEER_SETTINGS.env_type
    if env_type == "local":
        return QlibLocalEnv(conf=QlibLocalConf())
    if env_type == "conda":
        return QlibCondaEnv(conf=QlibCondaConf())
    if env_type == "docker":
        ensure_docker_supports_market(market_id, os.environ.get("RD_MARKET_PROVIDER_URI"))
        return QTDockerEnv(conf=QlibDockerConf())
    raise ValueError(f"Unknown env_type: {env_type}")


def generate_data_folder_from_qlib():
    """Run generate.py in a private build folder, then publish the two data folders.

    Nothing is written to the template folder, so a failed run cannot leave files behind
    that a later run, possibly for another market, would publish as its own data.
    """
    data_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder)
    debug_folder = Path(FACTOR_COSTEER_SETTINGS.data_folder_debug)
    for folder in (data_folder, debug_folder):
        folder.parent.mkdir(parents=True, exist_ok=True)
    env = _factor_data_env()
    env.prepare()
    # A conda env created by prepare() did not exist when its PATH was read, so read it again.
    if isinstance(env.conf, CondaConf) and not env.conf.bin_path:
        env.conf._update_bin_path()

    with tempfile.TemporaryDirectory(prefix=".factor_data_build_", dir=data_folder.parent) as build_dir:
        build = Path(build_dir)
        shutil.copy(FACTOR_DATA_TEMPLATE / "generate.py", build / "generate.py")
        result = env.run(entry="python generate.py", local_path=str(build))
        outputs = {data_folder: build / "daily_pv_all.h5", debug_folder: build / "daily_pv_debug.h5"}
        if result.exit_code != 0 or not all(path.is_file() for path in outputs.values()):
            # full_stdout, not stdout: stdout asks the LLM to filter the log.
            log_tail = result.full_stdout[-5000:]
            raise RuntimeError(f"Factor data generation failed (exit code {result.exit_code}). Log tail:\n{log_tail}")

        for target, output in outputs.items():
            staged = build / f"publish_{target.name}"
            staged.mkdir()
            shutil.move(output, staged / "daily_pv.h5")
            shutil.copy(FACTOR_DATA_TEMPLATE / "README.md", staged / "README.md")
            # Remove a leftover from an interrupted run so the move cannot nest or mix folders.
            if target.exists():
                shutil.rmtree(target)
            shutil.move(staged, target)


def refresh_readme(folder: Path) -> None:
    """Copy the template README into a published data folder when the two differ."""
    template = FACTOR_DATA_TEMPLATE / "README.md"
    target = folder / "README.md"
    if not target.is_file() or target.read_bytes() != template.read_bytes():
        shutil.copy(template, target)


def get_market_prompt_note(env: Mapping[str, str] | None = None) -> str | None:
    """One line telling the LLM which market it researches, or None where the prompts already fit.

    The prompts were written for China (SH/SZ codes), so cn-region runs get no note and keep
    their text byte for byte.
    """
    env = os.environ if env is None else env
    if current_market_id(env) is None or env.get("RD_MARKET_REGION", "cn") == "cn":
        return None
    return T(".prompts:qlib_market_note").r(
        region=env["RD_MARKET_REGION"],
        universe=env["RD_MARKET_QLIB_MARKET"],
        benchmark=env["RD_MARKET_BENCHMARK"],
    )


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string("""
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
""")

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_colwidth", None)

        df_info = "### Data Structure\n"
        df_info += (
            f"- Index: MultiIndex with levels {df.index.names}\n"
            if isinstance(df.index, pd.MultiIndex)
            else f"- Index: {df.index.name}\n"
        )

        df_info += "\n### Columns\n"
        columns = df.dtypes.to_dict()
        grouped_columns = {}

        for col in columns:
            if col.startswith("$"):
                prefix = col.split("_")[0] if "_" in col else col
                grouped_columns.setdefault(prefix, []).append(col)
            else:
                grouped_columns.setdefault("other", []).append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            relevant_line = ", ".join(f"{col}: {columns[col]}" for col in variable_list if col in columns)
            df_info += relevant_line + "\n"
        else:
            df_info += "#### All Columns:\n"
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                header = "Other Columns" if prefix == "other" else f"{prefix} Related Columns"
                df_info += f"\n#### {header}:\n"
                random.shuffle(cols)
                line = ", ".join(f"{col}: {columns[col]}" for col in cols)
                df_info += line + "\n"

        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### Sample Data\n"
            df_info += f"Showing data for instrument {one_instrument}:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )

    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )

    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib()
    if current_market_id() is not None:
        # Market data folders are built once, so a fixed README would otherwise never reach the prompts.
        # Plain runs may point data_folder at hand-made data with its own README, so they keep theirs.
        for folder in (FACTOR_COSTEER_SETTINGS.data_folder, FACTOR_COSTEER_SETTINGS.data_folder_debug):
            refresh_readme(Path(folder))
    content_l = []
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)
