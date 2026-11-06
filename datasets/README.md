# Fixed datasets

These are the original files used for training, not regenerated samples.
The launcher selects them automatically for the corresponding task preset.

| Task | File or directory | Stored instances | Training instances |
| --- | --- | ---: | ---: |
| JSSP10 | `jssp10/train.npy` | 100 | 100 |
| JSSP15 | `jssp15/train.npy` | 100 | 100 |
| JSSP20 | `jssp20/train.npy` | 100 | 100 |
| TSP50 | `tsp50/train.pkl` | 64 | 64 |
| CVRP50 | `cvrp50/train.pkl` | 64 | 64 |
| Max-Cut BA200 weighted | `maxcut/` | 100 | First 30 sorted filenames |

JSSP files are the `generatedData{size}_{size}_Seed200.npy` datasets from
[Offline-LD](https://github.com/jesserem/Offline-LD), with shape
`(100, 2, jobs, machines)` for processing times and machine assignments.
TSP50 is EoH's `examples/tsp_construct/evaluation/trainingdata/instances.pkl`,
containing coordinates and distance matrices. See the root [license](../LICENSE)
for the upstream notices.

CVRP50 is our fixed 64-instance training set (data seed 2024), with 50 customers,
vehicle capacity 50, coordinates, depot and demands. Max-Cut contains our original
100 weighted BA200 training graphs, stored as SciPy sparse `.npz` matrices.

`--seed` controls evolution, not data generation or subset selection. Additional
problem sizes require an explicit `--data` path. Generated programs are not
included here.

## Held-out test sets

`testing/` contains the original evaluation files, separate from training data.

| Evaluation preset | Instances | File under `testing/` |
| --- | ---: | --- |
| `jssp10`, `jssp15`, `jssp20` | 100 each | `jssp/generatedData{size}_{size}_Seed300.npy` |
| `jssp30x20`, `jssp50x20` | 100 each | `jssp/generatedData{jobs}_20_Seed300.npy` |
| `taillard20x20`, `taillard30x20` | 10 each | `jssp/tai{jobs}x20.npy` |
| `tsp20`, `tsp50`, `tsp100`, `tsp200` | 100 each | `tsp/instance_data_{size}.pkl` |
| `tsplib70` | 70 | `tsp/TSPlib_70instances.txt` |
| `cvrp50` | 100 | `cvrp/test.pkl` |
| `cvrp200` | 128 | `cvrp/vrp200_test_lkh.txt` |
| `vrplib192` | 192 | `cvrp/vrplib_192instances.txt` |

JSSP random tests use Offline-LD's Seed300 files; the Taillard arrays come from
the TBGAT benchmark distribution. TSP20/50/100/200 are the EoH testing files.
TSPLIB70, CVRP200 and VRPLIB192 are the LEHD/NCO_code evaluation files. CVRP50
is our fixed held-out set. Coordinates, demands and reference routes are kept
unchanged, including the native TSPLIB and VRPLIB coordinate scales.

Max-Cut's seven held-out graph sets are not bundled yet. The graphs in `maxcut/`
are training data, not substitutes for the test sets.

Evaluate the final training-selected program without further evolution:

```bash
python scripts/evaluate.py --dataset tsp200 --result-dir runs/tsp50/s0
```

Use `--dataset` to select any preset above. Generation 5 is the default.
For JSSP OOD tests, supply a JSSP20 training run; routing OOD tests use the
corresponding TSP50 or CVRP50 training run. No teacher or LLM is needed.

File sizes and SHA256 hashes are recorded in [`manifest.json`](manifest.json).
Verify the files from the repository root with:

```bash
python scripts/check_release.py
```
