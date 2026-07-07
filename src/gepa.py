import os
import shutil

from promptolution.optimizers.base_optimizer import BaseOptimizer
from promptolution.utils.logging import get_logger
from promptolution.utils.prompt import Prompt

try:
    from gepa.api import optimize as gepa_optimize
    from gepa.utils import MaxMetricCallsStopper
except (
    ImportError
):  # Optional baseline dependency; Tri-Fair itself does not require GEPA.
    gepa_optimize = None
    MaxMetricCallsStopper = None
from src.utils import generate_random_hash

logger = get_logger(__name__)


class BaseLLMAsGEPALM:
    """
    Wrap a promptolution BaseLLM to match GEPA's LanguageModel signature:
        Callable[[str], str]
    """

    def __init__(self, llm, system_prompt=None, seed=None):
        self.llm = llm
        self.system_prompt = system_prompt
        self.seed = seed

    def __call__(self, prompt):
        if self.seed is not None:
            self.llm.set_generation_seed(self.seed)
        return self.llm.get_response(prompt, system_prompts=self.system_prompt)[0]


class EvalOut:
    def __init__(self, outputs, scores, trajectories=None):
        self.outputs = outputs
        self.scores = scores
        self.trajectories = trajectories
        self.objective_scores: dict[str, float] | None = None


class PromptolutionGEPAAdapter:
    """
    Minimal GEPAAdapter backed by promptolution's BaseTask + BasePredictor.

    Candidate format:
      {"prompt": "<prompt text>"}
    """

    def __init__(
        self,
        task,
        predictor,
        system_prompts=None,
        component_name="prompt",
    ):
        self.task = task
        self.predictor = predictor
        self.system_prompts = system_prompts
        self.component_name = component_name
        self.propose_new_texts = None

    def evaluate(
        self,
        inputs,
        prog,
        capture_traces=False,
    ):
        xs = [x for x, _y in inputs]
        ys = [y for _x, y in inputs]

        preds, pred_seqs = self.predictor.predict(
            prompts=[prog[self.component_name]] * len(xs),
            xs=xs,
            system_prompts=self.system_prompts,
        )

        scores_np = self.task._evaluate(xs, ys, preds)
        scores = [float(s) for s in scores_np.tolist()]
        outputs = [str(s) for s in pred_seqs]
        trajectories = None
        if capture_traces:
            trajectories = []
            for idx, (x, y, pred, pred_seq, score) in enumerate(
                zip(xs, ys, preds, pred_seqs, scores)
            ):
                trajectories.append(
                    {
                        "trace_id": idx,
                        "input": x,
                        "target": y,
                        "prediction": pred,
                        "pred_seq": pred_seq,
                        "score": score,
                    }
                )

        return EvalOut(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(self, candidate, eval_batch, components_to_update):
        if not eval_batch or not eval_batch.trajectories:
            return {self.component_name: []}

        traces = list(eval_batch.trajectories)

        records = []
        for trace in traces:
            records.append(
                {
                    "Inputs": {"input": trace.get("input")},
                    "Generated Outputs": {
                        "prediction": trace.get("prediction"),
                        "raw": trace.get("pred_seq"),
                    },
                    "Feedback": f"Target: {trace.get('target')}",
                    "score": trace.get("score"),
                    "trace_id": trace.get("trace_id"),
                    "prompt_text": candidate.get(self.component_name),
                }
            )

        dataset = {self.component_name: records}
        if components_to_update:
            dataset = {k: v for k, v in dataset.items() if k in components_to_update}
        return dataset


class Gepa(BaseOptimizer):
    def __init__(
        self,
        predictor,
        task,
        initial_prompts=None,
        meta_llm=None,
        run_dir=None,
        config=None,
        callbacks=None,
        step_after_k_evals=1000,
    ):
        if gepa_optimize is None or MaxMetricCallsStopper is None:
            raise ImportError(
                "GEPA is an optional baseline dependency. Install the pinned gepa package "
                "before selecting optimizer='GEPA'."
            )
        super().__init__(
            predictor=predictor,
            task=task,
            initial_prompts=initial_prompts,
            callbacks=callbacks,
            config=config,
        )

        if run_dir is None:
            run_dir = os.path.join(
                "gepa_runs",
                f"{generate_random_hash()[:8]}",
            )

        self.run_dir = run_dir

        assert meta_llm is not None, "GEPA optimizer requires a reflection_llm"
        self.meta_llm = meta_llm
        self.reflection_lm_callable = BaseLLMAsGEPALM(
            llm=self.meta_llm,
        )

        self.step_after_k_evals = step_after_k_evals
        self.stopper = MaxMetricCallsStopper(max_metric_calls=step_after_k_evals)

    def _pre_optimization_loop(self):
        os.makedirs(self.run_dir, exist_ok=True)
        if os.listdir(self.run_dir):
            logger.warning(f"Cleaning existing GEPA run_dir={self.run_dir}")
            shutil.rmtree(self.run_dir)
            os.makedirs(self.run_dir, exist_ok=True)

        self.seed_candidate = self.prompts[0].construct_prompt()
        self.trainset = list(zip(self.task.xs, self.task.ys))

        self.gepa_adapter = PromptolutionGEPAAdapter(
            task=self.task,
            predictor=self.predictor,
            component_name="prompt",
        )

    def _step(self):
        result = gepa_optimize(
            run_dir=self.run_dir,
            seed_candidate={
                "prompt": self.seed_candidate
            },  # ignored if run_dir is non empty
            adapter=self.gepa_adapter,
            trainset=self.trainset,
            reflection_lm=self.reflection_lm_callable,
            stop_callbacks=[self.stopper],
        )

        self.scores = result.val_aggregate_scores
        candidates = result.candidates
        candidates = [prompt for d in candidates for prompt in d.values()]

        self.prompts = [Prompt(p) for p in candidates]

        self.stopper.max_metric_calls += self.step_after_k_evals
        return self.prompts
