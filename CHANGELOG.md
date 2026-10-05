# Changelog

## 0.1.0

First release. The CUDA-Q interface moves here from the Qrisp repository.

- `cudaq_kernel` turns a Qrisp function into a CUDA-Q kernel for `cudaq.run` or,
  with `execution_mode="sample"`, for `cudaq.sample`.
- `to_quake_mlir` lowers a Jaspr to Quake MLIR.
- `FixedShapeNDArray` annotates array parameters of a kernel.
