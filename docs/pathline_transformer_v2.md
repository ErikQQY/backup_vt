# PathlineTransformerV2

`PathlineTransformerV2` extends V1 without changing the V0 or V1
implementations. V1 already computes 21 trajectory/cross descriptors, but its
KNN attention only receives them indirectly through the node embedding. V2
uses those descriptors explicitly when computing attention weights.

## Physics-aware edge attention

For KNN edge \(i \rightarrow j\) and attention head \(h\), V2 adds

\[
b_{ijh} =
\tanh(g_h)
\left[
\frac{\langle W^q_h p_i, W^k_h p_j\rangle}{\sqrt{r}}
+ \operatorname{MLP}_h(R_{ij})
\right]
\]

to the original V1 attention logit. Here \(p_i\) contains the normalized V1
physics descriptors and \(R_{ij}\) contains:

- signed and absolute descriptor differences;
- pairwise products for LAVD, tangential velocity, local/deviation angular
  velocity, vorticity, swirl strength, and Q criterion;
- spatial/temporal separation;
- same-pathline, same-cross, and same-time indicators.

The result lets the model favor neighbors that are both geometrically close
and physically coherent. The bias is clipped before softmax for numerical
stability.

## Physics-aware temporal attention

V2 also learns a head-wise temporal weight for every pathline from its latent
feature and physics state. The attended feature is a gated residual over V1's
mean-plus-max temporal pool, so V1's behavior remains available throughout
training.

Both physics gates initialize to zero. After loading a V1 checkpoint, V2's
initial output is therefore equal to V1 (verified to absolute error
`0.000e+00` by `validate_pathline_transformer_v2.py`).

## Training

Warm-starting from the best V1 checkpoint is recommended:

```bash
python train.py \
  --config config/segmentation/pathline_transformer_v2.yaml \
  --data_dir /path/to/vatistas_trainable \
  --model_path /path/to/v1/best_checkpoint.pth.tar
```

For a from-scratch comparison using the V1 schedule:

```bash
python train.py \
  --config config/segmentation/pathline_transformer_v2.yaml \
  --data_dir /path/to/vatistas_trainable \
  --learning_rate 0.0001 \
  --epochs 200
```

Run the synthetic forward/backward, padding, registration, determinism, and
V1 warm-start regression checks with:

```bash
python validate_pathline_transformer_v2.py
```

## Evaluation

Use the new `TestPathlineSegmentation` task from the V2 config. It aggregates
TP/FP/FN/TN over every pathline and reports accuracy, balanced accuracy,
precision, recall, F1, and IoU.

The legacy `TestSegmentation` task indexes `prediction[..., 0]` and therefore
evaluates only the first pathline of a `[B,K]` output. The previously logged
87% and 90% values are precision under that legacy protocol, not full
pathline-level accuracy. Re-evaluate V0, V1, and V2 with the same fixed split,
seed, threshold, and `TestPathlineSegmentation` task before claiming an
accuracy improvement.

## Suggested ablations

Keep the data split and seed fixed and compare:

1. V1 checkpoint with both V2 gates at zero;
2. edge physics attention only (`temporal_gate_init: 0.0`, then freeze the
   temporal-pool parameters);
3. temporal physics attention only (freeze each
   `physics_logit_scale`);
4. the complete V2 model;
5. `physics_attention_heads` in `{4, 8}` and `k` in `{16, 24}`.

Select checkpoints by validation loss or validation F1, then test exactly
once. Report at least F1 and IoU alongside accuracy because the foreground
pathline labels are imbalanced.
