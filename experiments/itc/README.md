# ITC 2019 benchmark inputs

The GitHub Actions workflow automatically downloads three official ITC 2019
test instances: `lums-sum17`, `wbg-fal10` and `pu-cs-fal07`. They are public
copies from the competition organization's
[solver repository](https://github.com/itc2019/alexandre-lemos/tree/c33d15797686a27c192eabb90948baa54d3ddef5/data/input/ITC-2019).
Downloads use that fixed commit and are checked against recorded SHA-256 hashes.
The upstream license is retained in `UPSTREAM_LICENSE.txt`.

The full suite also generates `synth-s`, `synth-m` and `synth-l`. These are
**synthetic ITC-style instances**, labelled separately from the official data
in `benchmark-metadata/itc-manifest.json`. The smoke suite generates `synth-s`
and `synth-m`; its performance tests select the two smallest available XML files.

To add another instance, put its original problem XML in this directory and
commit it. The workflow discovers additional XMLs automatically. Use safe
filenames containing letters, digits, dots, underscores and hyphens. The
`synth-*` filenames are reserved for generated data.

Every dataset is passed through the existing **adaptation A**, rather than
the complete ITC 2019 competition model. The manifest includes the fidelity
report describing merged weeks, the time grid and omitted constraints.

The full suite runs **H1, H5 and H9** on each prepared ITC dataset, matching
the ITC scenarios already covered by the performance smoke tests. S/M/L/XL
continue to run H1-H9. Initial timetables are generated once and shared
between benchmark jobs. ITC datasets use one seed; S/M use three and L/XL one.

Preparation records ITC datasets for which no starting timetable was found,
and ITC scenarios that cannot be constructed, with their reasons in
`benchmark-metadata/benchmark-plan.json`. These are preparation or scenario
limitations, not solver proofs of infeasibility. The full suite requires at
least one official and one generated ITC dataset to be prepared successfully.

After committing a workflow change, start a **new** Actions run and choose
`full`. A run already in progress continues using its original commit.
