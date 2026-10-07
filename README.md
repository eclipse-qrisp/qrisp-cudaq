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

## Documentation

The CUDA-Q tutorial and API reference are part of the Qrisp documentation at
[qrisp.eu](https://www.qrisp.eu).

## License

[EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0](LICENSE)
