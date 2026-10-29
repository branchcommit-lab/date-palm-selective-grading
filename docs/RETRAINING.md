# Training-method resources

Original model-development implementations are provided as source notebooks under `src/methods/`. They document the baseline grader, synthetic appearance modelling, Dandhi-targeted refinement, photographic perturbations, representation restrictions, cultivar holdouts and excluded-size-class experiments.

```bash
python run.py retrain
```

This command records the optional training-method registry; it does **not** train new models. Repeating training requires obtaining the source photographs, appropriate ImageNet initializations, experiment-specific configurations, and computational resources. Numerical results reproduced by the Results workflow do not require retraining.
