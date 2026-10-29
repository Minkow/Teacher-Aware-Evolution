# Data and pretrained teachers

Datasets and pretrained weights are not included in the source distribution.
Obtain them from their original projects and follow their licenses. Keep them
under ignored `data/`, `checkpoints/`, and `external/` directories.

Teacher source projects:

- [Offline-LD](https://github.com/jesserem/Offline-LD) for JSSP.
- [NCO_code / LEHD](https://github.com/CIAM-Group/NCO_code) for routing.
- [ECO-DQN](https://github.com/tomdbar/eco-dqn) for Max-Cut.

| Task | `--data` | `--teacher-root` | `--checkpoint` |
| --- | --- | --- | --- |
| JSSP | Training `.npy` array | Offline-LD source root | Offline-LD checkpoint directory |
| TSP | Coordinate dataset `.pkl` | LEHD `single_objective` directory | TSP `.pt` checkpoint |
| CVRP | Depot/customer/demand `.pkl` | LEHD `single_objective` directory | CVRP `.pt` checkpoint |
| Max-Cut | Directory of benchmark `.npz` graphs | ECO-DQN source root | ECO-DQN `.pth` checkpoint |

The routing adapter expects the LEHD directory layout containing `LEHD/TSP`
and `LEHD/CVRP`. Default upstream checkpoint names are `checkpoint-150.pt`
(TSP) and `checkpoint-40.pt` (CVRP); the launcher accepts explicit paths.
Use the matching teacher architecture and install its upstream dependencies.
JSSP checkpoint directories must contain the files expected by Offline-LD's
`best` checkpoint loader.

Only load pickle datasets and model checkpoints from trusted sources.

## Initial programs

Pass an external JSON list of five records using `--seed-programs`. Each record
contains `algorithm` (a description) and `code` (Python implementing the task
interface). Historical generated programs are not included in this package.
Use the same initialization across methods for controlled comparisons.

The launcher provides experiment presets. Native entry points expose problem
dimensions and training counts through their `--help` options; evaluators consume
the supplied instances rather than identifying them by benchmark name.

