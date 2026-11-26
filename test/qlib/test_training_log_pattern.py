import re

import pytest

from rdagent.scenarios.qlib.experiment.workspace import TRAINING_LOG_PATTERN

pytestmark = pytest.mark.offline


def test_training_log_keeps_positive_losses():
    # Lines as pytorch_general_nn writes them for an MSE loss, between lines that are not kept.
    log = "\n".join(
        [
            "[1:MainThread](2026-11-26 16:44:34,452) INFO - qlib.GeneralPTNN - [a.py:299] - Epoch0:",
            "[1:MainThread](2026-11-26 16:46:31,347) INFO - qlib.GeneralPTNN - [a.py:305] - Epoch0: train 0.992726, valid 0.996557",
            "[1:MainThread](2026-11-26 16:48:35,046) INFO - qlib.GeneralPTNN - [a.py:305] - Epoch1: train 0.991044, valid 0.997792",
            "[1:MainThread](2026-11-26 16:48:35,048) INFO - qlib.GeneralPTNN - [a.py:327] - best score: 0.996557 @ 0 epoch",
            "[1:MainThread](2026-11-26 16:48:36,451) INFO - qlib.GeneralPTNN - [a.py:344] - Test samples: 363600",
        ]
    )

    assert re.findall(TRAINING_LOG_PATTERN, log) == [
        "Epoch0: train 0.992726, valid 0.996557",
        "Epoch1: train 0.991044, valid 0.997792",
        "best score: 0.996557 @ 0 epoch",
    ]


@pytest.mark.parametrize(
    "line",
    [
        "Epoch3: train -0.998100, valid -0.999200",
        "Epoch0: train nan, valid nan",
        "Epoch12: train 1.5e-05, valid -inf",
        "best score: -0.999200 @ 3 epoch",
        "best score: -inf @ 0 epoch",
    ],
)
def test_training_log_keeps_other_scores(line):
    assert re.findall(TRAINING_LOG_PATTERN, f"INFO - {line}\n") == [line]
