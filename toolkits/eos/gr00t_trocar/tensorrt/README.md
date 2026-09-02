# GR00T N1.7 TensorRT Backbone

This directory owns the RLInf-side qualification contract for the GR00T N1.7
TensorRT frozen-backbone hybrid. The online implementation remains Python and
uses TensorRT's Python runtime with PyTorch CUDA tensors. Praxis and Poiesis are
design references only and are not runtime, build, or artifact dependencies.

The implementation is qualified in two distinct phases:

1. Reproduce the official B1 LIBERO seven-engine pipeline at Isaac-GR00T commit
   `51d4c89f72fda44cbf77285c6a8114b52676b8a1`.
2. Build and integrate a true-static-B8 Trocar backbone containing only the ViT
   and LLM TensorRT engines. The VLLN, action head, and value head stay in
   PyTorch and remain hot-updateable.

The official oracle is not the RLInf performance baseline. The matched RLInf
comparison is:

- control: PyTorch frozen backbone plus eager PyTorch action/value head;
- candidate: TensorRT frozen backbone plus the same eager PyTorch head.

Both arms enable the existing unused-logits optimization and disable rollout
feature reuse, making the backbone executor the only behavioral variable.

Large ONNX and TensorRT artifacts are not stored in Git. Unqualified build
outputs remain under the attempt directory. A bundle is promoted to the EOS
artifact cache only after manifest, deserialize, binding, numerical, and stream
interop gates pass.

The normative machine-readable contract is
`contract-n1d7-trocar-b8.json`. Any contract change must be reviewed before an
EOS qualification result can be retained.
