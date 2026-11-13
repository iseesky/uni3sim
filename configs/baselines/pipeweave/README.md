# PipeWeave hardware adaptation

The external PipeWeave calculators and pretrained models are shared with the original checkout. Adaptations belong in `src/kernel/predictors/pipeweave_adapter.py`. The paper comparison explicitly selects the NVIDIA pretrained model with `model_hardware="A100"` on all three platforms.

## Ascend resource mapping

The hardware counts and capacities follow `microbench/ascend_910b/references/Ascend910B2.ini`, which contains platform configuration rather than measured performance. GEMM and MoE projections use 24 Cube cores. Pure Vector operators use 48 AIVs and 192 KiB UB per AIV. The Cube profile retains 512 KiB L1 per core. Both views retain the same chip-wide HBM and L2 bandwidth fields.

Attention keeps the upstream 24-group tile selection and total work. Each group's padded Vector work is divided between two AIVs. The total XU cycle feature uses 48 AIVs, while the maximum-worker XU work and cycle features describe one member of the pair. Matrix and memory features retain their original Cube-group interpretation. This is a mapping of the baseline's padded tile schedule, not an exact simulation of the CANN scheduler.

## Vector arithmetic parameters

The theoretical FP32 capacities come from Huawei's [Vector instruction performance table](https://asc.gitcode.com/api/appendix/vector_instruction_theoretical_perf_summary.html), in its Atlas A2/A3 section. The reference was checked on 2026-10-20.

| PipeWeave field | Value | Interpretation |
|---|---|---|
| `FmaFp32` | 64 | FP32 element operations per cycle per AIV for Add, Mul and FusedMulAdd |
| `xuFp32` | 32 | FP32 element operations per cycle per AIV for the Exp, Ln, Sqrt and Div instruction class |

The 32 value denotes primitive instruction throughput. It does not denote 32 complete reciprocal-square-root results per cycle. A serial Sqrt followed by Div has a theoretical rate of 16 logical results per cycle. That composite rate must not be substituted into an upstream count of primitive operations without changing its meaning.

The upstream RMSNorm and SiLU calculators retain their aggregate GPU-derived instruction inventories, including their existing multipliers and feature conventions. In particular, the RMSNorm calculator does not expose an opcode-level inventory. Applying the Vector profile to it remains a baseline approximation. The 40 key-0 RMSNorm cases that use a scalar expression and handoffs in CANN remain represented by PipeWeave's original uniform Vector abstraction. No claim of source-exact scalar modeling is made.

## Scope and limitations

No measured mechanism latency, including the proposed 50 ns scalar-normalization measurement, is read or added by this adaptation. No production latency is used to choose parameters, and no model is retrained. Numerical failures remain failures and are reported without clipping utilization or substituting another predictor.

This revision changes the core-domain mapping and arithmetic capacities. The existing `memBandwidth`, `l2CacheBandwidth` and `sharedMemoryBandwidth` values remain inherited baseline parameters. Their correspondence to all CANN memory paths is not established by this change. The adapter is a resource-level transfer of PipeWeave, not a complete native CANN analytical model.
