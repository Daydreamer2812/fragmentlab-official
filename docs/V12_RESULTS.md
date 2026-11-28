# Shardelix V12 release evidence

V12 retains V11's classical CPU-only blind framework and expands its usable
settings and its missing/continuous recovery. All comparisons below use
independent evaluation after solving; no reference image, true coordinates,
angle labels or cut template enters the solver. These are particular measured
cases, not universal accuracy guarantees.

## New algorithms and profiles

- Multi-gap thin-strip solver: internal long-axis correlations estimate
  discontinuity distance; reciprocal endpoint uniqueness, global missing
  budget and graph consistency permit a candidate chain or abstention.
- Irregular continuous masks: independent two-wall support anchors a relative
  component to the frame; bounded relative-group recovery uses verified
  contacts; geometric budget and observable objective decide acceptance.
- Unknown-width, arbitrary-angle digital document strips: common height and
  rectangular alpha moments infer width and rotation, then affine normalization
  and the document ordering route. Cannot measure missing total width or infer
  a reliable order from indistinguishable blank edges.
- Fast reduces refinement, Balanced preserves the default path, Accurate
  verifies additional hypotheses. All use one public Solver and the same
  cut-family kernel routes. Accurate can consume more work without improving
  the result. Native/wasm may differ when native-only MILP is applicable.
- A shared, bounded single/batch workflow is used by CLI-facing framework
  and both Studios. The evaluator is separate; no synthetic truth in blind mode.

## Paired recovery holdouts

Run `PYTHONPATH=src python benchmarks/benchmark_v12.py --help` for reproduction
against the V11 source tree. The cases below were not used to develop the
multi-gap algorithm. Both versions solved exactly the same shuffled inputs.
Strict pass = DPA and adjacency F1 at least 95%, with no collisions.

| Test | V11 pass | V12 pass |
|---|---:|---:|
| Multi-gap strip holdouts (8) | 0/8 | 4/8 |
| Quarter-turn subset (5) | 0/5 | 3/5 |
| Continuous subset (3) | 0/3 | 1/3 |

Four successful cases recover every present strip (DPA/F1 100%). Four remain
ambiguous/incorrect. Missing areas are left unfilled. See every row including
failures in reports/v12/recovery.csv. The solver does not force a plausible
placement when confidence or budget is insufficient.

## Native scale and cut-family retest

Synthetic texture seed 2719, 256 pieces, quarter-turn, all seven cut families:
DPA/F1 100%, zero collisions. Other scale tests use independent seed 11226.

| Test | DPA | F1 | Placed | Seconds | RSS MiB | Gate |
|---|---:|---:|---:|---:|---:|---|
| Grid 16,384, 20px tiles | 100% | 100% | 16,384 | 27.75 | 328 | Pass |
| Grid 16,384, 8px tiles | 0% | 10.50% | 1,161 | 34.04 | 299 | Fail |
| Grid 65,536, 12px tiles | 99.85% | 99.86% | 65,438 | 199.47 | 864 | Pass |
| Jigsaw 256 generated, 10% missing, continuous/native, seed 9217 | 91.30% | 92.26% | 214/230 | 17.11 | 120 | Fail |

All four cases have zero collision pixels. The last case has identical DPA/F1
to V11; it does not demonstrate a recovery improvement. Grid gauge alignment
can remove global orientation ambiguity for scoring, not infer absolute orientation.
V11's 65k and V10's 262k measurements are historical on different inputs.
Do not compare runtimes of different tile sizes directly or claim 262k retested.
Stress jobs overlapped in this environment, so time and RSS are descriptive,
not controlled speedup estimates. See reports/v12/stress.csv and cases/*.json.

## Performance profile matrix

12 cases: grid/jigsaw × 64/256 × Fast/Balanced/Accurate, quarter-turn seed 42.
11/12 meet the strict gate. The 256-piece jigsaw Fast places 17 only (DPA 0%,
F1 6.34%); Balanced and Accurate place 256, both 100% DPA/F1.
Solver times ~0.48 s, 4.97 s, 8.20 s respectively on this run. More work
does not guarantee improvement; timing not isolated and batch peak RSS is
process high-water. Full matrix: reports/v12/profiles.csv.

## Unknown-width continuous document strips

Four new synthetic cases, 12 strips each, two seeds on both texture and
text-like documents. Common height and every individual width inferred
correctly in all four; absolute angle MAE under 0.001 degree. Zero collisions.
Texture order DPA 100% (both); text-like DPA 33.33% and 8.33%. Ordering remains
ambiguous on blank/repeated document edges. Results in
reports/v12/document_continuous.csv. Not a real-world paper dataset claim.

## Verification

The V12 research snapshot passed 61 algorithm tests and 9 workflow tests.
This curated public edition retains all 61 algorithm tests and the web build
regression; internal tooling tests and logs are intentionally excluded.
The same engine also passed real Chromium/WebAssembly CI after two responsive
layout fixes: single solve, batch sweep, blind replay, mobile viewport and
zero JavaScript errors. Public CI runs these checks independently.
A new 64-piece synthetic jigsaw showcase is measured in
reports/v12/showcase.json and reproduced by scripts/make_showcase.py.

## Remaining scientific limitations

Missing irregular native cases below 95%, underconstrained gaps near ends,
short/flat/repeating edges, real torn paper, perspective, unknown scale,
unobservable total missing strip width and highly ambiguous maps/LSU remain
open. No claim of unique solution when there is insufficient observable data.
Large browser capacity needs actual device tests. V12 numerical evidence is included. Historical private logs, older previews and third-party data are intentionally excluded.
