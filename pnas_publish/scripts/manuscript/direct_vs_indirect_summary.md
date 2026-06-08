# Prior-Boundary Diagnostic Summary

This is a minimal synthetic diagnostic, not a final benchmark. The direct-inversion results test whether an amortized learned inverse mapping tends to return structures toward the support of its training prior when the target model is near or outside that support.

Run command: `scripts/eval_prior_boundary_effect.py --device cpu --n-test 128 --n-envelope 512 --posterior-samples 16 --sampling-steps 16 --batch-size 8 --n-forward-eval 30 --forward-inv-steps 120 --indirect-uncertainty-samples 3 --indirect-multistarts 4 --indirect-uncertainty-steps 60`

## Main Metrics

| Method | Test set | n | Vs MAE | Vs RMSE | Disp MAE | Coverage | Target outside | Pred inside given outside | Runtime (s) | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| DI-Strong | in-prior | 128 | 0.0769 | 0.1251 | 0.0304 | 0.5464 | 0.0164 | 0.7917 | 3.4262 | ok |
| DI-Strong | boundary | 128 | 0.3149 | 0.4307 | 0.2080 | 0.2599 | 0.6098 | 0.5415 | 3.3545 | ok |
| DI-Strong | out-of-prior | 128 | 0.6699 | 0.8724 | 0.2345 | 0.1543 | 0.7362 | 0.6940 | 3.3890 | ok |
| DI-Weak | in-prior | 0 | NA | NA | NA | NA | NA | NA | NA | skipped_missing_checkpoint |
| DI-Weak | boundary | 0 | NA | NA | NA | NA | NA | NA | NA | skipped_missing_checkpoint |
| DI-Weak | out-of-prior | 0 | NA | NA | NA | NA | NA | NA | NA | skipped_missing_checkpoint |
| IND-FWD | in-prior | 30 | 1.8806 | 2.2302 | 2.2701 | NA | 0.0342 | 0.4347 | 21.3498 | ok |
| IND-FWD | boundary | 30 | 1.7394 | 2.0453 | 1.3857 | NA | 0.5907 | 0.2768 | 20.8000 | ok |
| IND-FWD | out-of-prior | 30 | 1.8833 | 2.1932 | 1.0689 | NA | 0.7289 | 0.2788 | 20.5875 | ok |
| IND-FWD-uncertainty | in-prior | 3 | 1.1029 | 1.4214 | NA | 0.3806 | NA | NA | 4.1916 | ok |
| IND-FWD-uncertainty | boundary | 3 | 1.3455 | 1.5970 | NA | 0.3937 | NA | NA | 4.2206 | ok |
| IND-FWD-uncertainty | out-of-prior | 3 | 1.9466 | 2.2442 | NA | 0.4249 | NA | NA | 4.1812 | ok |

## Conservative Interpretation

1. DI-Strong should be interpreted primarily as a posterior surrogate under the synthetic prior and simulator used during training. Its in-prior behavior is the relevant baseline.
2. Boundary and out-of-prior rows diagnose whether predictions are pulled back toward the strong-prior support. A large `pred_inside_given_target_outside` indicates prior-boundary collapse rather than successful extrapolation.
3. DI-Weak is only comparable if a weak-prior checkpoint is supplied or trained. A skipped row means the current project did not contain that checkpoint at run time.
4. IND-FWD is a preliminary control-point inversion through a forward surrogate. It may reduce the prior-boundary bias of direct inverse mappings, but it remains dependent on the surrogate training domain, control-point parameterization, optimization bounds, initialization, and regularization.
5. Any result intended for the paper should be rerun with larger test sets, more posterior samples, an independently trained weak-prior model, and a validated forward surrogate.

## Paper-Ready Use

The strongest paper-facing result from this diagnostic is the prior-support caveat: amortized direct inversion can be accurate inside the synthetic prior while becoming biased near or outside the prior support. The weak-prior and forward-surrogate rows should be presented as preliminary unless they are rerun at larger scale.
