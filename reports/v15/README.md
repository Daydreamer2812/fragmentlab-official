# V15 paired benchmark archive

- [Physical DocDataset scans](docdataset_real.json): real1/2/3, seed 624.
- [Mechanical strips](mechanical.json): 30 previously observed and seven new held-out cases.
- [Scan perturbations](scan_stress.json): eight deterministic synthetic cases, with order, orientation and row-pose metrics.
- [Continuous eroded fragments](damage.json): four deterministic small cases.
- [Blind DAFNE development](dafne_development.json): one difficult external simulated-artwork case, not a successful reconstruction.

See [methods and acceptance criteria](../../docs/TECHNICAL_REPORT.md).
The same data and protocol were used for paired inference. Time measurements
are descriptive because background development jobs overlapped; none are
used to assert a controlled end-to-end speedup. Input datasets are not
included. Obtain them from their [original sources](../../docs/DATASETS.md).
