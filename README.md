# qrisp-cudaq

CUDA-Q backend for [Eclipse Qrisp](https://github.com/eclipse-qrisp/Qrisp). It compiles
Qrisp programs to the Quake MLIR dialect of [CUDA-Q](https://nvidia.github.io/cuda-quantum/)
and runs them as native CUDA-Q kernels.

## Installation

```bash
pip install qrisp[cudaq]
```

This installs Qrisp together with `qrisp-cudaq`. CUDA-Q supports Linux (x86_64 and
aarch64) and macOS on Apple silicon (CPU simulation only), with Python 3.11 or later.
Windows is not supported.

## Simulation targets

CUDA-Q is built for GPU-accelerated simulation and for quantum hardware. On a
machine with an NVIDIA GPU, CUDA-Q uses its
[GPU state-vector simulator](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html#cuquantum-single-gpu)
by default. The [CUDA-Q backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/backends.html)
page lists all simulators and
[hardware backends](https://nvidia.github.io/cuda-quantum/latest/using/backends/hardware.html).

Without an NVIDIA GPU, for example on macOS, CUDA-Q falls back to its CPU simulator
[`qpp-cpu`](https://nvidia.github.io/cuda-quantum/latest/using/backends/sims/svsims.html#qpp-cpu-backend),
which the CUDA-Q documentation describes as good "for basic testing and
experimentation with just a few qubits". Expect it to become slow from around 20
qubits, in particular with `cudaq.run`, which simulates the kernel again for every
shot.

If a kernel does not use any measurement result inside the kernel,
`@cudaq_kernel(execution_mode="sample")` together with `cudaq.sample` simulates it
once and returns the counts of the measured bitstrings for all shots. Kernels that
use measurement results inside the kernel need the default `execution_mode="run"`.
This includes much of Qrisp's arithmetic, which uncomputes auxiliary qubits with
mid-circuit measurements.

## Example

```python
import cudaq
from qrisp import QuantumVariable, cx, h, measure
from qrisp_cudaq import cudaq_kernel


@cudaq_kernel
def bell():
    qv = QuantumVariable(2)
    h(qv[0])
    cx(qv[0], qv[1])
    return measure(qv)


print(cudaq.run(bell, shots_count=100))
```

## Documentation

The CUDA-Q tutorial and API reference are part of the Qrisp documentation at
[qrisp.eu](https://www.qrisp.eu).

## License

[EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0](LICENSE)
