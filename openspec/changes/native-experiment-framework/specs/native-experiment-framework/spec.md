## ADDED Requirements

### Requirement: Native transmission and lawful evidence
The framework SHALL transmit native RGB with a grid independent of classifier input and account for every transmitted coded bit. Receiver evidence MUST exclude simulation truth.
#### Scenario: Native image through three baselines
- **WHEN** a 32x32 image is sent with uniform, IAQ or addressed transmission
- **THEN** reconstruction stays 32x32, classification uses one shared differentiable adapter, and each budget is respected including headers and redundancy.
#### Scenario: Soft decoding
- **WHEN** AWGN or ideal-CSI Rayleigh is simulated
- **THEN** physical soft observations, masks and numeric uncertainty are available without ground-truth errors as inputs.

### Requirement: Reproducible experiment lifecycle
The framework SHALL save deterministic disjoint splits, configuration and source fingerprints, resumable training and evaluation, and paired sample records.
#### Scenario: Interrupted task
- **WHEN** an identical experiment resumes
- **THEN** complete training epochs and evaluated sample batches are retained without duplicating records; incompatible configurations are rejected.
#### Scenario: Condition ablation
- **WHEN** repair variants are compared under the same method and channel seed
- **THEN** they receive the same damaged image and capacity-controlled evidence inputs.
#### Scenario: Frozen final evaluation
- **WHEN** a completed training run is evaluated on the official test pool or extrapolated SNRs
- **THEN** its verified best checkpoints are reused without training or model selection, and results record their source run.
#### Scenario: Explicit channel shift
- **WHEN** the user explicitly enables channel-shift evaluation
- **THEN** only the channel family and receiver noise-variance mismatch may additionally change; model structure and all other protocol parameters remain compatible.
#### Scenario: Known classifier exposure
- **WHEN** a locally fitted classifier or its recorded ancestors used images now assigned to calibration, audit or test
- **THEN** the experiment is rejected before fitting or transmission; unknown external pretraining exposure remains disclosed as unknown.

### Requirement: Documentation and safe migration
The project SHALL document setup, smoke and formal usage, outputs, extensions and limitations, and remove superseded active scripts after backup verification.
#### Scenario: Legacy code removal
- **WHEN** obsolete experiment files are removed
- **THEN** their exact contents and hashes remain in a migration archive while datasets, weights, documents and previous experimental outputs are preserved.

### Requirement: Honest reporting
The framework MUST distinguish synthetic workflow checks from trained-model research and summarize uncertainty by image rather than treating noise repeats as independent samples.
#### Scenario: Smoke report
- **WHEN** a synthetic or random-classifier run completes
- **THEN** reports clearly label its purpose and do not present it as an innovation performance result.
#### Scenario: Large result matrix
- **WHEN** an experiment contains millions of repeated image observations
- **THEN** summary generation streams prediction records and retains aggregate image statistics instead of the full prediction table in memory.
