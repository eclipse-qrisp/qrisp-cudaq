# Changelog

## 0.2.0

- Require CUDA-Q 0.17 (`cudaq>=0.17.0,<0.18`); support for CUDA-Q 0.16 is dropped.

## 0.1.0

First release. The CUDA-Q interface moves here from the Qrisp repository.

- `cudaq_kernel` turns a Qrisp function into a CUDA-Q kernel for `cudaq.run` or,
  with `execution_mode="sample"`, for `cudaq.sample`.
- `to_quake_mlir` lowers a Jaspr to Quake MLIR.
- `FixedShapeNDArray` annotates array parameters of a kernel.
