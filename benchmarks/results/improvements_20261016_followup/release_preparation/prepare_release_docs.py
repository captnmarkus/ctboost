"""Edit only the isolated release checkout; do not build or publish docs."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

STUDY = Path(r"C:\apps\ctboost")
RELEASE = Path(r"C:\Users\captn\.codex\worktrees\release-0-1-62\ctboost")
PREP = STUDY / "benchmarks/results/improvements_20261016_followup/release_preparation"
FILES = ["README.md", "docs/index.md", "docs/releases.md", "mkdocs.yml", "docs/benchmarks.md"]
before = {name: (RELEASE / name).read_bytes() for name in FILES}
active = {name: (STUDY / name).read_bytes() for name in FILES}
text = {name: raw.decode("utf-8").replace("\r\n", "\n") for name, raw in before.items()}
sha = lambda raw: hashlib.sha256(raw).hexdigest()
archive = "https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/improvements_20261016_followup"

def replace(name, old, new):
    assert text[name].count(old) == 1, (name, old[:80])
    text[name] = text[name].replace(old, new)

name = "README.md"
start = text[name].index("The current release,")
end = text[name].index("\n## Highlights", start)
text[name] = text[name][:start] + '''The [0.1.62 release](https://captnmarkus.github.io/ctboost/release-0.1.62/)
reduces CPU prediction preparation costs for eligible categorical pandas inputs
and wide fitted schemas, writes compact JSON, and improves the precision of
categorical target statistics. Conditional feature testing still selects a
feature before its cut point is optimized; learning defaults remain unchanged.
Newly fitted pipelines use format5 for precise CTR accumulation. Formats1–4
remain loadable with their original arithmetic; older CTBoost versions cannot
load format5. See the release notes for compatibility and validation scope.
''' + text[name][end:]
start = text[name].index("The [0.1.61 release notes]", text[name].index("## Evidence and project status"))
end = text[name].index("\nFor opt-in CPU learning controls", start)
text[name] = text[name][:start] + f'''The [October follow-up archive]({archive}) separates production correctness,
fixed-model inference measurements, and a twelve-task development accuracy
study. It retains failed hypotheses, resource limits, source identities, and
the distinction between model training and inference runtimes. These local
studies do not establish a new TabArena Elo or a universal accuracy advantage.

<!-- RELEASE_PREPARATION: replace this pending sentence after final quality and timing receipts. -->
The final full-data round-count CV and release serving-cost comparison are
pending; their terminal results will be recorded in the archive before release.

The [0.1.61 verification archive](https://github.com/captnmarkus/ctboost/tree/833a775/benchmarks/results/release_readiness_20260908)
and [earlier accuracy studies](https://github.com/captnmarkus/ctboost/tree/8cc75aa/benchmarks/results/inference_accuracy_20260908/accuracy)
remain historical evidence. The status recorded on **9 September 2026** was
that a fresh0.1.61 Full HPO25 evaluation had been prepared but had not started,
and the0.1.60 run was retired. That record is not a completed0.1.62 evaluation.
The published author-run0.1.58 TabArena-Lite bundle is also separate from this
release and is not an official leaderboard entry. The
[benchmark page](https://captnmarkus.github.io/ctboost/benchmarks/) retains its
scores, resource limits, and historical comparisons.
''' + text[name][end:]

name = "docs/index.md"
start = text[name].index("The current release,")
end = text[name].index('\n!!! info "Project status"', start)
text[name] = text[name][:start] + '''[CTBoost0.1.62](release-0.1.62.md) improves categorical CPU inference
preparation, compact JSON persistence, and CTR accumulation precision.
Conditional-inference feature selection and learning defaults are preserved.
Newly fitted pipelines use format5; existing format1–4 pipelines retain their
legacy arithmetic. Read the release notes before exchanging new artifacts with
an older CTBoost runtime.
''' + text[name][end:]
start = text[name].index("As of 9 September 2026,")
end = text[name].index('\n<div class="grid cards"', start)
text[name] = text[name][:start] + f'''The [October follow-up]({archive}) records the local inference and twelve-task
accuracy work, including unsuccessful experiments. It is separate from the
historical0.1.58 author-run TabArena bundle. The9 September2026 statement that
the0.1.61 Full HPO25 run was prepared but unstarted remains a dated historical
status, not a result for0.1.62. See [benchmark status](benchmarks.md) for the
protocols, pending final study stages, and limits.
''' + text[name][end:]

name = "docs/releases.md"
start = text[name].index("The current release is")
end = text[name].index("Use the current", start)
text[name] = text[name][:start] + '''[CTBoost0.1.62](release-0.1.62.md) covers categorical prediction preparation,
compact JSON persistence and precise CTR accumulation. Its release notes state
the artifact compatibility and validation scope.
''' + text[name][end:]
replace(name, "| [0.1.61](release-0.1.61.md)", "| [0.1.62](release-0.1.62.md) | Categorical serving, compact JSON, and CTR precision |\n| [0.1.61](release-0.1.61.md)")
replace("mkdocs.yml", "      - Current release (0.1.61): release-0.1.61.md", "      - Release 0.1.62: release-0.1.62.md")

name = "docs/benchmarks.md"
start = text[name].index("## Current public-evidence status")
end = text[name].index("### CTBoost 0.1.58:", start)
text[name] = text[name][:start] + f'''## CTBoost0.1.62: local evidence and scope

[Release0.1.62](release-0.1.62.md) reduces prediction preparation overhead,
improves categorical target-statistic precision, and writes compact JSON.
The conditional-inference feature-test-before-cut design and learning defaults
remain unchanged. The [October follow-up archive]({archive}) records the
production changes, tests, fixed-model latency protocols and accuracy studies.

The accuracy study contains twelve tasks: four binary, four regression and
four multiclass. It compares CTBoost, XGBoost and LightGBM with explicit fitting,
selection and replication budgets. Development data were reused across
successive hypotheses; their scores are development evidence, not unbiased
estimates or an official TabArena ranking. Equal model-fit counts do not imply
equal compute or model capacity. The immutable accuracy runtime is recorded
separately from the final inference runtime; the CTR precision fix does not
retroactively change the training of those frozen models.

<!-- RELEASE_PREPARATION: replace this pending paragraph after final quality and timing receipts. -->
The final full-current-data round-count CV is in progress. Its frozen policy
requires five fold-local fits at each of three round counts per task/library,
followed by mandatory full-data fits. Conditional second-seed and official-test
stages remain governed by the original quality gates. The final serving-cost
comparison waits for all permitted quality work to finish. No completed result
for those pending stages is claimed here.

Serving-cost comparisons use one complete, prospectively selected model policy
and balanced library orders. They distinguish raw-input preparation from
normalized-input prediction, and require exact prediction checks. Their task
scope and differing model capacities must accompany any reported speedup;
latency alone does not establish accuracy superiority.

## Historical public evaluations

The [0.1.61 accuracy archive](https://github.com/captnmarkus/ctboost/tree/8cc75aa/benchmarks/results/inference_accuracy_20260908/accuracy)
retains earlier local studies and failed qualification gates. Those studies
did not establish an Elo increase or justify changing learning defaults.

The status recorded on **9 September2026** was that a fresh Full HPO25
evaluation of public **0.1.61** was prepared but had **not started**. Its design
covered the default plus25 frozen HPO configurations across816 outer splits,
with eight-fold bagging. The0.1.60 run was retired. This is a historical status
record, not a completed0.1.62 Full evaluation or an official leaderboard entry.

On8 September2026, Lennart Purucker separately
[reported TabArena-Lite results with25 configurations and offered a Full run with200](https://github.com/autogluon/tabarena/pull/479#issuecomment-5588951890).
That report predates0.1.61 and is not evidence for0.1.62.

The [0.1.60 CTR and numeric-bin audit](https://github.com/captnmarkus/ctboost/tree/master/benchmarks/results/score_audit_20260907)
and earlier grouped-statistic panels remain historical development evidence,
with their own protocols and limitations.

''' + text[name][end:]
# Preserve the numerical historical bundle but avoid making a newly verified
# claim that it is still the latest available artifact on an external site.
replace(name, "The latest [downloadable author-run result bundle]", "The historical [downloadable author-run result bundle]")

# Human-facing prose should use normal spaces around numbers/versions.
for name in FILES:
    for old, new in (("CTBoost0.", "CTBoost 0."), ("Release0.", "Release 0."),
        ("fresh0.", "fresh 0."), ("the0.", "the 0."), ("completed0.", "completed 0."),
        ("run0.", "run 0."), ("historical0.", "historical 0."), ("for0.", "for 0."),
        ("format5", "format 5"), ("Formats1", "Formats 1"), ("format1", "format 1"),
        ("the9 ", "the 9 "), ("The9 ", "The 9 "), ("The0.", "The 0."), ("September2026", "September 2026"),
        ("plus25", "plus 25"), ("across816", "across 816"), ("On8 ", "On 8 "),
        ("with25", "with 25"), ("with200", "with 200"), ("predates0.", "predates 0.")):
        text[name] = text[name].replace(old, new)

files = []
for name in FILES:
    assert (RELEASE / name).read_bytes() == before[name]
    backup = PREP / "docs_before" / name
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_bytes(before[name])
    destination = RELEASE / name
    temp = destination.with_name(destination.name + ".release-docs.tmp")
    temp.write_text(text[name], encoding="utf-8", newline="\n")
    temp.replace(destination)
    assert (STUDY / name).read_bytes() == active[name]
    files.append(dict(path=name, before_sha256=sha(before[name]), after_sha256=sha(destination.read_bytes())))
receipt = dict(recorded_at=datetime.now(timezone.utc).isoformat(), status="isolated_release_docs_prepared_pending_final_study_results",
    release_root=str(RELEASE), files=files, active_study_docs_unchanged=True,
    tests_docs_build_commits_or_pushes=False, pending_before_release=[
        "Replace the two RELEASE_PREPARATION blocks with terminal18b/allowed later-stage and final timing evidence.",
        "Update the home-page reference to pending study stages when the terminal outcome is known.",
        "Root-owned release0.1.62 notes/version surfaces and final CI/publishing remain separate work."],
    historical_status="0.1.61 FullHPO prepared/not-started is explicitly dated9September2026; no completed official0.1.62 result claimed")
(PREP / "docs_preparation.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
print(json.dumps(dict(receipt=str(PREP / "docs_preparation.json"), sha256=sha((PREP / "docs_preparation.json").read_bytes()), files=len(files))))
