# Fixed training datasets

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
problem sizes require an explicit `--data` path. Held-out evaluation datasets and
generated programs are not included here.

File sizes and SHA256 hashes are recorded in [`manifest.json`](manifest.json).
Verify the files from the repository root with:

```bash
python scripts/check_release.py
```
