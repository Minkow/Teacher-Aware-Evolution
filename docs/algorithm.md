# Algorithm and configuration

The release uses the Inline Soft-Anchor implementation. The paper names its
operators structural rewrite (`s`), parameter calibration (`p`), and
mechanism-level fusion (`m`). These identifiers are used throughout the code,
configuration, analyzer schema, and output records. Child schedules are configured
with `EOH_S_CHILD_SCHEDULE`, `EOH_P_CHILD_SCHEDULE`, and `EOH_M_CHILD_SCHEDULE`.

Structural rewrite and parameter calibration receive the generation-level
population diagnosis. Mechanism-level fusion reads only its
two selected parents' code, objective values, self traces, and on-policy teacher
evidence. It diagnoses the pair inside the generation request; it does not read
the population analyzer's E guidance or issue another pair-analysis request.
Better is a performance reference, not a mandatory code backbone.

Population management considers objective, alignment, and measured decision
runtime when available. Exact pairs are sampled without replacement and each
Better parent can appear in at most two fusion requests per batch.

| Setting | JSSP | TSP50 | CVRP50 | Max-Cut |
| --- | --- | --- | --- | --- |
| Performance slack | 2% | 5% | 5% | 2% |
| Generation-final runtime multiplier | 5 for JSSP20; disabled for JSSP10/15 | Disabled | Disabled | 5 |
| Teacher state budget | 20 | 20 | 20 | 20 |
| Population size | 12 | 12 | 12 | 12 |

Performance slack is a task default in the population manager, not a launch
argument. Each run records the applied value in `run_config.json`.

Runtime maintenance occurs after all three operators. It removes final-population
members above five times that population's minimum positive measured runtime,
then fills vacancies with eligible candidates. It does not reject slow children
at generation time.

All four tasks record decision runtime. Evaluation uses the same worker pipeline
for one or multiple workers. CVRP runtime measurement is a release correction:
it activates the runtime axis that was absent from the original serial evaluator.

The source manifest identifies the frozen source used for this release. The
Full population manager is restored from its reference snapshot rather than
using the selection-alignment ablation manager. Release changes include explicit
asset and credential configuration, action validation, dataset-size accounting,
and unified worker evaluation. Teacher-enabled objective evaluation uses the
same metadata path as population evaluation.

Generated code executes in reusable, supervised rollout processes, including
JSSP diagnostic scoring. An evaluation deadline covers queueing, worker startup,
and execution. On timeout, queued work for that candidate is cancelled and its
active workers are killed and reaped without cancelling other candidates.
Standalone evaluators use `EOH_EVALUATION_TIMEOUT_S` (default 1800 seconds);
evolution supplies the remaining evaluation/FE budget instead. Max-Cut's
per-instance limit is also enforced by the parent process. Teacher inference
remains in the main process; these controls are not a security sandbox.

