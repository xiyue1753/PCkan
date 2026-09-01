"""DeepSeek-based HPO agent for the joint hyperparameter + augmentation study.

Two modes:
  - with_context=True  (groups a/b): receives model mechanism facts, data
    statistics, search-space schema and numeric history -> mechanism-aware
    joint proposals with explicit reasoning (JSON "analysis" field).
  - with_context=False (group g, R2#6 context-isolation): receives ONLY the
    same numeric history format as the classical baselines (config + objective)
    plus the bare schema; no model code, mechanism notes or data statistics.

Every request/response is appended to a JSONL log (never the API key).
Temperature, model and endpoint come from config; temperature is fixed at
LLM_TEMPERATURE (0.1) for the main runs and swept in E8.
"""
import os, sys
_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_DIR = os.path.dirname(_LLM_DIR)
sys.path.insert(0, _LLM_DIR)
sys.path.insert(0, _PROJ_DIR)
os.chdir(_PROJ_DIR)          # cnn_lstm_kan/ keeps s_data/ paths working

import json
import os
import time

from openai import OpenAI

from config import (DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, LLM_TEMPERATURE,
                    LLM_MAX_TOKENS, get_deepseek_api_key)
from llm_search_space import validate, space_schema_str

MODEL_CODE = """MODEL CODE (complete; the model being tuned, use_kanconv=True branch):
model = CNN_LSTM_Head(
    in_features=7,                 # input features: GHI/DHI/temperature/angles/time...
    hidden_size=<hidden_dim>,      # searchable: LSTM hidden size
    num_layers=<num_layers>,       # searchable: LSTM layer count
    out_channels=<out_channels>,   # searchable: conv channels
    horizon=24,                    # forecast horizon: next 24 steps
    head_type='FC',                # output head: Linear(hidden, 24)
    use_kanconv=True, kan_kernel=3, kan_grid=3, kan_k=3,
    kan_gate_init=1.0,             # learnable gate on the spline deviation
    pool_mode='max',               # MaxPool1d(2)
    dropout=<dropout>)             # searchable: shared Dropout prob (after conv and LSTM)

Forward path with tensor shapes:
  x (B,24,7) -> KANConv1d: per-channel cubic B-spline calibration
               (identity-initialized, grid 3, k=3) + linear conv k=3,pad=1
               -> (B, out_channels, 24)
            -> BatchNorm1d -> ReLU
            -> Conv1d(out_channels->out_channels, k=3, pad=1) -> BN -> ReLU
            -> MaxPool1d(2) -> (B, out_channels, 12)
            -> Dropout(dropout)
            -> LSTM(out_channels -> hidden_size, num_layers, batch_first)
               -> (B, 12, hidden_size), last step taken
            -> Dropout(dropout)
            -> Linear(hidden_size, 24) -> (B, 24)

Training protocol (environment facts):
  - 15-min data, 24-step lookback -> 24-step forecast; MinMax(-1,1) normalization
  - Loss: MSE; Optimizer: Adam(lr); Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)
  - FIXED training budget (no early stopping; run a fixed epoch count);
    full-batch gradient descent (batch size 4096)
  - Validation set = a chronological prefix of the series (25% of the training
    windows) - a realistic "new plant" scenario: the training distribution is
    limited and seasonally skewed
  - Augmentation applies to the normalized input windows (B,24,7) only; labels
    y (B,24) are never changed."""

MECHANISM_NOTES = """MECHANISM UNDERSTANDING (qualitative, about this model family; no empirical claims):
- The first layer is KANConv1d: a learnable per-channel B-spline calibration
  (identity-initialized, grid 3, cubic) followed by a linear conv3. The
  calibration starts as the identity and learns a conservative adjustment of
  each input channel's value mapping; the learnable gate (init 1.0) rescales
  the spline deviation and can shrink when the identity calibration suffices
  (self-regularizing).
- Gains of such calibrations tend to concentrate where the irradiance -> power
  mapping is most nonlinear (mid/high GHI); the LSTM is the load-bearing
  component for the temporal structure.
- Perturbing inputs while keeping labels unchanged (e.g. input noise alone or
  masking) creates contradictory supervision for the spline calibration that
  is learning the GHI->power signal; label-consistent strategies (mixup family)
  do not break that signal. This is a property of the mechanism, not a
  recommendation - which strategy is best for the given data is an empirical
  question to probe from the data statistics."""

AUG_SEMANTICS = """AUGMENTATION PARAMETER SEMANTICS (definitions only, no selection
advice; augmentation applies to normalized input windows [-1,1]): the four
strategies are INDEPENDENT - each has its own switch (use_*) and its own
sigma. You may enable ANY subset at once: a single strategy, several
strategies mixed together, or (in non-forced settings) none. Nothing is
pre-combined; the composition is your decision.
- mixup:          convex combination of two (x, y) pairs:
                  x_mix = w*x1 + (1-w)*x2, y_mix = w*y1 + (1-w)*y2, w~Beta(a,a).
                  Labels mix consistently with inputs (safe for regression).
                  sigma_mixup = Beta shape a (small a: near-pure samples /
                  strong mixing; a=0.5: smoother mixing).
- mixup_phys:     same mixing, but pairs restricted to the same GHI-state
                  bin (terciles of window-mean GHI) - mixed samples keep the
                  GHI->power relationship physically plausible (random
                  pairing can create impossible interpolations, e.g. sunny
                  power with cloudy GHI). sigma_phys = Beta shape a.
- mixup_temporal: same mixing, but pairs restricted to the same hour-of-day
                  bin (peak-position proxy) - keeps the intraday phase
                  plausible. sigma_temporal = Beta shape a.
- jitter_ghi:     small Gaussian noise on the IRRADIANCE channels only
                  (GHI, +DHI if present) - the physical drivers of PV power;
                  angular/time features stay untouched. Labels unchanged.
                  sigma_jitter = noise std on the [-1,1] scale.
- augmentation_factor: training set multiplier round(factor) (total training
                  samples = original samples x factor); directly increases
                  the training cost.
- When several strategies are enabled they compose in sequence per copy
  (mixing first, then jitter). Which subset and intensities are best for the
  given data is an empirical question for you to probe."""

DESIGN_PRINCIPLES = """DESIGN PRINCIPLES (reason from these; no per-parameter selection guidance):
1. Data characteristics <-> augmentation strategy matching: choose strategy
   and intensity from the data statistics (sample size, feature ranges,
   variance structure), not from assumptions.
2. Balance between model capacity and augmentation intensity: avoid
   augmentation so strong that learning becomes difficult.
3. Synergy between augmentation and model parameters: some capacity/lr
   configurations make specific augmentation strategies more effective.
4. Computational efficiency: augmentation_factor directly scales the
   training cost.
5. With limited data, the model should be neither overly complex nor overly
   simple - match capacity to the amount of data.
6. With a FIXED training budget, a too-large learning rate destabilizes
   training and a too-small one leaves the model underfit within the budget
   - pick a reasonable value within the given bounds."""

SEARCH_DISCIPLINE = """SEARCH DISCIPLINE (mandatory):
1. Explore: never propose the same (or nearly identical) configuration twice.
   If the requested config was already evaluated, propose a clearly different
   region of the space.
2. Diversity first: early iterations should span the space; refine only after
   a few diverse candidates have been measured.
3. Do not anchor on the first working configuration - compare across at least
   two distinct capacity/lr regions before committing.
4. Bayesian-style iteration (when a history of previous iterations is
   provided): synthesize the current best configuration, reason about WHY it
   is the best, analyze why the failed configurations failed, and correct
   course accordingly.
5. Different parameters have different impact magnitudes: avoid adjustments
   too small to matter (wasted trials) or so large that they overshoot.
6. Initial proposal: be aggressive - the goal is a strong starting point,
   not a conservative one."""

HISTORY_DIAGNOSIS = """DIAGNOSTIC FRAMEWORK (apply when a history of previous iterations is
provided - use validation objective and, when visible, the training trend):
1. High validation error with low training error -> overfitting: increase
   regularization/augmentation strength or adjust capacity.
2. Both validation and training errors high -> underfitting: reduce
   augmentation, adjust learning rate or capacity.
3. Augmentation ineffective or barely changing results -> try a different
   strategy or adjust model parameters.
4. First confirm the training error is being effectively optimized: if
   optimization itself is failing, synergy analysis is meaningless.
5. Optimize primarily with model parameters, augmented secondarily by the
   augmentation strategy."""

RESPONSE_EXAMPLE = """RESPONSE FORMAT (respond ONLY with this JSON, no extra text):
{
  "analysis": "explicit reasoning: the data characteristics / mechanism /
               historical patterns you considered, and how the chosen
               augmentation subset + intensities interact with the model
               configuration (1-3 sentences)",
  "config": {
    "learning_rate": <value within bounds>,
    "hidden_dim": <multiple of 16 within bounds>,
    "num_layers": <value within bounds>,
    "out_channels": <multiple of 16 within bounds>,
    "dropout": <value within bounds>,
    "use_mixup": <true or false>,
    "use_mixup_phys": <true or false>,
    "use_mixup_temporal": <true or false>,
    "use_jitter": <true or false>,
    "sigma_mixup": <value within bounds>,
    "sigma_phys": <value within bounds>,
    "sigma_temporal": <value within bounds>,
    "sigma_jitter": <value within bounds>,
    "augmentation_factor": <value within bounds>
  }
}"""


def build_system_prompt(with_context, space):
    if not with_context:
        return ("You are a hyperparameter optimizer for a time-series forecasting "
                "model. You will receive a search space schema and a numeric "
                "history of previously evaluated configurations (each with its "
                "validation MSE). Propose the next configuration. Respond ONLY "
                "with JSON: {\"analysis\": \"<brief reasoning>\", "
                "\"config\": {<all searchable fields>}}.")
    return f"""You are an expert in deep learning, time-series forecasting, KAN-based
architectures and data augmentation, performing JOINT optimization of model
hyperparameters and input-window augmentation for a photovoltaic power
forecasting task.

{TASK_INTRO}

{MODEL_CODE}

SEARCH SPACE (the only fields that may be varied; every value MUST fall
within the given bounds - never assume values outside the bounds are valid):
{space_schema_str(space)}

{AUG_SEMANTICS}

{MECHANISM_NOTES}

{DESIGN_PRINCIPLES}

{SEARCH_DISCIPLINE}

{HISTORY_DIAGNOSIS}

Rules:
1. Only the fields listed in the search space may be varied; keep every
   other field at its current value (they are fixed).
2. Augmentation applies to the normalized input windows only (labels never
   change).
3. Every value must lie within the bounds given in the search space.

{RESPONSE_EXAMPLE}"""

TASK_INTRO = ("Task: propose hyperparameters + augmentation strategy for "
              "CNN_LSTM_Head(KANConv-M5) minimizing validation MSE. "
              "Augmentation parameters are optimized JOINTLY with the model "
              "hyperparameters, analyzing their synergistic effects.")


def build_user_prompt_initial(with_context, data_stats, space, base):
    fixed = {k: v for k, v in base.items() if k not in space}
    if not with_context:
        return (f"Search space schema (only these fields may be varied):\n"
                f"{space_schema_str(space)}\n\n"
                f"Fixed fields (do not change): {json.dumps(fixed, ensure_ascii=False)}\n\n"
                f"Data: {data_stats['samples']} training windows, "
                f"{data_stats['features']} input features, "
                f"normalized to [-1, 1].\n\n"
                "Propose the initial configuration (JSON with analysis + config).")
    return (f"Data statistics (site {data_stats['site']}):\n"
            f"- training windows: {data_stats['samples']}\n"
            f"- features: {data_stats['features']}\n"
            f"- feature ranges (min..max): {data_stats['ranges']}\n"
            f"- target (power) normalized to [-1,1] for training\n"
            f"- fixed fields (do not change): {json.dumps(fixed, ensure_ascii=False)}\n\n"
            "Propose the initial joint configuration, explicitly reasoning "
            "about augmentation/model coupling (JSON with analysis + config).")


def build_user_prompt_update(with_context, history, current_config, data_stats, space):
    if not with_context:
        lines = ["Numeric history (config -> validation MSE), lower is better:"]
        for h in history[-10:]:
            lines.append(f"  iter {h['trial']}: obj={h['objective']:.5f} "
                         f"cfg={json.dumps(h['config'], ensure_ascii=False)}")
        return ("\n".join(lines) + "\n\nOnly these fields may be varied: "
                + space_schema_str(space)
                + "\nPropose the next configuration (JSON with analysis + config).")
    lines = ["Previous iterations (validation MSE, lower is better):"]
    for h in history[-10:]:
        a = h.get('analysis') or ''
        lines.append(f"  iter {h['trial']}: obj={h['objective']:.5f} "
                     f"cfg={json.dumps(h['config'], ensure_ascii=False)}"
                     + (f" | LLM analysis: {a[:200]}" if a else ""))
    lines.append(f"\nCurrent best config:\n{json.dumps(current_config, ensure_ascii=False)}")
    lines.append("\nApply the diagnostic framework: assess convergence state "
                 "(overfitting/underfitting), explain why the best config is best "
                 "and why the failed configs failed, then propose the next joint "
                 "config (JSON with analysis + config).")
    return "\n".join(lines)


class LLMAgent:
    name = 'llm'

    def __init__(self, temperature=None, with_context=True, log_dir=None,
                 max_retries=2, model=None, base_url=None, api_key=None):
        self.temperature = LLM_TEMPERATURE if temperature is None else temperature
        self.with_context = with_context
        self.max_retries = max_retries
        self.model = model or DEEPSEEK_MODEL
        self.base_url = base_url or DEEPSEEK_BASE_URL
        self.api_key = api_key or get_deepseek_api_key()
        if not self.api_key:
            raise RuntimeError('DeepSeek API key not configured '
                               '(see config.get_deepseek_api_key)')
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        log_dir = log_dir or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          'llm_logs')
        os.makedirs(log_dir, exist_ok=True)
        tag = 'ctx' if with_context else 'noctx'
        self.log_path = os.path.join(log_dir, f'agent_{tag}_t{self.temperature}.jsonl')
        self.history = []

    # ------------------------------------------------------------------
    def _log(self, kind, payload):
        with open(self.log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'ts': time.time(), 'kind': kind,
                                'temperature': self.temperature,
                                'model': self.model, **payload},
                               ensure_ascii=False) + '\n')

    def _call(self, system, user, space, base):
        from config import DEEPSEEK_THINKING, LLM_REASONING_EFFORT
        for attempt in range(self.max_retries + 1):
            try:
                kwargs = dict(
                    model=self.model,
                    messages=[{'role': 'system', 'content': system},
                              {'role': 'user', 'content': user}],
                    temperature=self.temperature,
                    max_tokens=LLM_MAX_TOKENS,
                    response_format={'type': 'json_object'})
                if DEEPSEEK_THINKING:
                    kwargs['reasoning_effort'] = LLM_REASONING_EFFORT
                    kwargs['extra_body'] = {'thinking': {'type': 'enabled'}}
                resp = self.client.chat.completions.create(**kwargs)
                msg = resp.choices[0].message
                content = msg.content
                reasoning = getattr(msg, 'reasoning_content', None) or ''
                self._log('response', {'attempt': attempt, 'user': user,
                                       'raw': content,
                                       'reasoning': reasoning[-2000:]})
                cfg = self._parse(content, space, base)
                if cfg is not None:
                    return cfg, content
                self._log('parse_fail', {'attempt': attempt, 'raw': content})
                if attempt < self.max_retries:
                    user = user + ('\n\nYour previous answer was not valid JSON or '
                                   'violated the schema. Return ONLY '
                                   '{"analysis": "...", "config": {...}}.')
            except Exception as e:
                self._log('api_error', {'attempt': attempt, 'error': str(e)})
                if attempt < self.max_retries:
                    time.sleep(3)
        return None, None

    @staticmethod
    def _parse(content, space, base):
        """Parse the LLM JSON and return a validated config honoring the
        searched sub-space: fields outside `space` stay locked at `base`."""
        if not content:
            return None
        text = content.strip()
        if text.startswith('```'):
            text = text.strip('`')
            if text.startswith('json'):
                text = text[4:].strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find('{')
            end = text.rfind('}')
            if start < 0 or end <= start:
                return None
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        raw_cfg = data.get('config', {})
        if not isinstance(raw_cfg, dict):
            return None
        merged = dict(base)
        for k, v in raw_cfg.items():
            if k in space:
                merged[k] = v
        config = validate(merged)
        if config is None:
            return None
        return {'analysis': str(data.get('analysis', ''))[:500], 'config': config}

    # ------------------------------------------------------------------
    def propose(self, history=None, data_stats=None, space=None, base=None,
                feedback=None):
        """Return {'analysis':..., 'config':...} or None.

        history: list of trial records (as produced by llm_runner.run_search).
        space: sub-space to search (MODEL_ONLY_SPACE / AUG_ONLY_SPACE / None=full).
        base:  config whose non-searchable fields stay fixed (default
               DEFAULT_CONFIG; stage-2 of the separate group passes the best
               model config as base).
        feedback: optional string appended to the user prompt (used by the
                  dedup loop: 'this config was already evaluated with obj X;
                  propose a different one').
        """
        from llm_search_space import SPACE, DEFAULT_CONFIG
        space = space or SPACE
        base = base or DEFAULT_CONFIG
        h = history or []
        stats = data_stats or {'site': '?', 'samples': '?',
                               'features': '?', 'ranges': '?'}
        system = build_system_prompt(self.with_context, space)
        if len(h) == 0:
            user = build_user_prompt_initial(self.with_context, stats, space, base)
        else:
            best_cfg = min(h, key=lambda r: r['objective'])['config']
            user = build_user_prompt_update(self.with_context, h,
                                            best_cfg, stats, space)
        if feedback:
            user = user + f'\n\n[FEEDBACK] {feedback}'
        self._log('request', {'system': system, 'user': user})
        out, _ = self._call(system, user, space, base)
        if out is not None:
            self.history.append(out)
        return out

    def report(self, config, objective):
        pass
