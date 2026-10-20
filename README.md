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
from qrisp import QuantumFloat, h, measure
from qrisp_cudaq import cudaq_kernel


@cudaq_kernel
def main(k: int):
    a = QuantumFloat(3)
    h(a)  # a is in superposition of 0, 1, ..., 7
    b = QuantumFloat(4)
    b[:] = k
    b += a  # adds all eight values of a to k at once
    return measure(a), measure(b)


print(cudaq.run(main, 5, shots_count=5))
# [(1.0, 6.0), (3.0, 8.0), (0.0, 5.0), (4.0, 9.0), (1.0, 6.0)]
```

## Troubleshooting

### `LLVM ERROR: Option 'default' already exists!` on import

On Linux, importing `cudaq` and Qrisp in the same process can abort with this
message. It happens when `numba` and `llvmlite` were installed from conda-forge.
That build of `llvmlite` loads LLVM as a shared library, which clashes with the
LLVM inside CUDA-Q
([NVIDIA/cuda-quantum#2420](https://github.com/NVIDIA/cuda-quantum/issues/2420)).
Qrisp imports `numba`, so `import cudaq` followed by `import qrisp` is enough to
trigger it.

The `llvmlite` wheels on PyPI link LLVM statically and do not clash. Replace the
conda-forge packages with the PyPI wheels of the same versions, so that packages
depending on them keep working. With `numba` 0.60.0 and `llvmlite` 0.43.0
installed, that is:

```bash
conda list | grep -E "numba|llvmlite"  # shows the installed versions
pip install --force-reinstall --no-deps numba==0.60.0 llvmlite==0.43.0
python -c "import cudaq, numba"  # aborts before the fix, succeeds after
```

## Documentation

The CUDA-Q tutorial and API reference are part of the Qrisp documentation at
[qrisp.eu](https://www.qrisp.eu).

## License

[EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0](LICENSE)
