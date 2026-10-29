# Inference interface and inputs

The default inference mode checks the previously generated prediction evidence without executing image inference:

```bash
python run.py inference
```

To inspect required external inputs, run:

```bash
python run.py inference --action preflight
```

Fresh image inference is a separate operation:

```bash
python run.py inference --action execute
```

The execution driver requires the source images, appropriate preprocessing inputs, checkpoints and split/benchmark manifests. Required arguments include `--study-inputs`, `--feature-inputs`, `--appearance-checkpoint`, `--calibration-manifest`, `--development-manifest` and, where applicable, `--path-map`. The driver reports missing files and exits without inference if its input requirements are not satisfied. Source images are available from the dataset publisher; the original research model weights are not included in this package.

The default evidence-audit mode reproduces the statistical analyses from recorded predictions. It is not a second run of raw-image inference.
