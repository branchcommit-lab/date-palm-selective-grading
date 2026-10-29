# Research-data release scope

The repository distributes observation-level predictions, recorded experimental configurations, statistical methods, numerical verification utilities, published result tables, and figure-regeneration code.

The source-group registry identifies photographs by their recorded source identities and SHA-256 digests; the photographic image data are obtained from the original dataset source. Notebook paths are local configuration inputs, with no fixed institutional file locations. Actual image-level inference requires the corresponding external images, checkpoints and mask inputs.

All numerical evaluations in Results mode use the packaged experiment records. The code does not rerun training or produce new predictions without separate image and model-weight inputs.
