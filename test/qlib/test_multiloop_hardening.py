"""Behaviour that only matters from the second loop of a run on, or with a small LLM."""

import pandas as pd
import pytest

import rdagent.scenarios.qlib.proposal.quant_proposal as quant_proposal
from rdagent.components.coder.CoSTEER import CoSTEER
from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.core.experiment import Experiment, Task
from rdagent.log import rdagent_logger
from rdagent.scenarios.qlib.developer.feedback import decision_from_llm
from rdagent.scenarios.qlib.proposal.bandit import extract_metrics_from_experiment
from rdagent.scenarios.qlib.proposal.quant_proposal import (
    QlibQuantHypothesisGen,
    QuantTrace,
)

pytestmark = pytest.mark.offline


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("yes", True),
        ("No", False),
        (True, True),
        ("Yes.", True),
        ("no, keep the SOTA result", False),
        ("replace it", False),  # not a clear yes or no
        ("yes or no", False),  # the answer template, echoed
        ("<true or false>", False),
        ("No, yes it should replace", False),
        (None, False),
        (1, False),
    ],
)
def test_an_unclear_replace_decision_counts_as_no_instead_of_ending_the_run(answer, expected):
    assert decision_from_llm(answer) is expected


def test_the_bandit_reads_the_annualized_return():
    class Finished:
        result = pd.Series(
            {
                "IC": 0.0155,
                "1day.excess_return_with_cost.annualized_return": 0.0713,
                "1day.excess_return_with_cost.max_drawdown": -0.1625,
            }
        )

    metrics = extract_metrics_from_experiment(Finished())

    assert metrics.arr == pytest.approx(0.0713)
    assert metrics.sharpe == pytest.approx(0.0713 / 0.1625)


def test_the_bandit_reward_falls_with_a_deeper_drawdown_and_ignores_missing_metrics():
    from rdagent.scenarios.qlib.proposal.bandit import EnvController, Metrics

    controller = EnvController()
    shallow = Metrics(ic=0.01, arr=0.07, mdd=-0.10)
    deep = Metrics(ic=0.01, arr=0.07, mdd=-0.40)
    missing = Metrics(ic=float("nan"), arr=0.07, mdd=-0.10)

    assert controller.reward(shallow) > controller.reward(deep)
    assert controller.reward(missing) == pytest.approx(controller.reward(Metrics(arr=0.07, mdd=-0.10)))


def test_an_action_the_llm_invents_falls_back_to_factor(monkeypatch):
    class Answering:
        def build_messages_and_create_chat_completion(self, *args, **kwargs):
            return '{"action": "both"}'

    monkeypatch.setattr(quant_proposal.QUANT_PROP_SETTING, "action_selection", "llm")
    monkeypatch.setattr(quant_proposal, "APIBackend", Answering)
    scen = FakeScenario()
    gen = QlibQuantHypothesisGen(scen)

    gen.prepare_context(QuantTrace(scen))

    assert gen.targets == "factor"


class FakeScenario:
    market_prompt_note = None


@pytest.mark.parametrize("action", ["factor", "model"])
def test_a_fixed_action_is_taken_in_every_loop(monkeypatch, action):
    monkeypatch.setattr(quant_proposal.QUANT_PROP_SETTING, "action_selection", action)
    scen = FakeScenario()
    gen = QlibQuantHypothesisGen(scen)

    context, _ = gen.prepare_context(QuantTrace(scen))

    assert gen.targets == action
    assert ("GRU" in context["RAG"]) is (action == "model")


def test_an_unknown_action_selection_is_refused(monkeypatch):
    monkeypatch.setattr(quant_proposal.QUANT_PROP_SETTING, "action_selection", "factors")
    scen = FakeScenario()

    with pytest.raises(ValueError, match="Unknown action_selection: 'factors'"):
        QlibQuantHypothesisGen(scen).prepare_context(QuantTrace(scen))


def test_the_hypothesis_keeps_the_decided_action_when_the_llm_echoes_another(monkeypatch):
    monkeypatch.setattr(quant_proposal.QUANT_PROP_SETTING, "action_selection", "factor")
    scen = FakeScenario()
    gen = QlibQuantHypothesisGen(scen)
    gen.prepare_context(QuantTrace(scen))

    hypothesis = gen.convert_response('{"hypothesis": "20-day momentum", "reason": "trend", "action": "model"}')

    assert hypothesis.action == "factor"


class OneTask(Task):
    def get_task_information(self) -> str:
        return self.name


class FakeWorkspace:
    def create_ws_ckp(self) -> None:
        pass

    def recover_ws_ckp(self) -> None:
        pass


class OneStepStrategy:
    def evolve_iter(self, *, evo, evolving_trace, queried_knowledge):
        evo.sub_workspace_list = [FakeWorkspace()]
        yield evo


class PassingEvaluator:
    def evaluate_iter(self, *, evolving_trace, queried_knowledge):
        yield  # kick-off
        yield True  # the evolved step
        return CoSTEERMultiFeedback(
            [CoSTEERSingleFeedback(execution="ok", return_checking="ok", code="ok", final_decision=True)]
        )


class CountingRag:
    """Stands in for CoSTEERRAGStrategyV2, whose generate_knowledge skips steps below its counter."""

    def __init__(self) -> None:
        self.current_generated_trace_count = 0
        self.calls = []

    def query(self, evo, evolving_trace):
        return None

    def load_dumped_knowledge_base(self) -> None:
        pass

    def dump_knowledge_base(self) -> None:
        pass

    def generate_knowledge(self, evolving_trace) -> None:
        self.calls.append((self.current_generated_trace_count, len(evolving_trace)))
        self.current_generated_trace_count = len(evolving_trace)


def test_each_develop_generates_knowledge_from_its_first_step(monkeypatch):
    monkeypatch.setattr(rdagent_logger, "log_object", lambda *args, **kwargs: None)
    coder = CoSTEER(settings=CoSTEERSettings(), eva=PassingEvaluator(), es=OneStepStrategy(), scen=None)
    rag = CountingRag()
    coder.rag = rag

    coder.develop(Experiment(sub_tasks=[OneTask("first")]))
    coder.develop(Experiment(sub_tasks=[OneTask("second")]))

    # (steps already turned into knowledge, steps in the trace). The second develop used to see (1, 1)
    # and skip its only step.
    assert rag.calls == [(0, 1), (0, 1)]
