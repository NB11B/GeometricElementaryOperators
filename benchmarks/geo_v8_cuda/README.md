# GEO V8 CUDA Full-Cycle Benchmark

This directory provides the first implementation stage for issue #33. It builds independently from the host-only top-level CMake project so CUDA can be validated without changing the accepted CPU build.

## Implemented in this stage

- CUDA reference forward path that recomputes geometric-product signs on device;
- CUDA planned-batch forward path using the host-generated routing/sign plan;
- CUDA parameter VJP;
- residual/loss fusion plus VJP and SGD update path;
- correctness matrix for dimensions 2-6, all canonical signatures, left/right action, and batches 1/16/64/256;
- native CUDA benchmark with resident, transfer+compute, and end-to-end timing classes;
- PyTorch eager CUDA and `torch.compile` CUDA comparators;
- repeated-trial aggregation with median, min, max, mean, standard deviation, MAD, and raw JSON.

The hand-written CUDA comparator and profiler automation remain separate follow-up gates. No GPU performance claim is accepted until physical correctness, repeated trials, profiler evidence, and artifact hashing are complete.

## Windows build

From the repository root in a PowerShell session where `nvcc` is available:

```powershell
cmake -S .\benchmarks\geo_v8_cuda `
    -B .\build\geo-v8-cuda `
    -G "Visual Studio 17 2022" `
    -A x64 `
    -DCMAKE_BUILD_TYPE=Release `
    -DCMAKE_CUDA_ARCHITECTURES=120

cmake --build .\build\geo-v8-cuda `
    --config Release `
    --parallel
```

If the installed CMake/CUDA pair rejects architecture `120`, omit `-DCMAKE_CUDA_ARCHITECTURES=120` and allow `native` detection, then preserve the generated CMake cache in the evidence bundle.

## Correctness gate

```powershell
ctest `
    --test-dir .\build\geo-v8-cuda `
    -C Release `
    --output-on-failure
```

Direct execution:

```powershell
& .\build\geo-v8-cuda\Release\test_v8_cuda_correctness.exe
```

Acceptance marker:

```text
GEO_V8_CUDA_CORRECTNESS: PASS
```

Do not benchmark after a failed correctness gate.

## Single native CUDA benchmark trial

```powershell
New-Item -ItemType Directory -Force .\gpu-results | Out-Null

& .\build\geo-v8-cuda\Release\bench_v8_cuda.exe `
    .\gpu-results\geo-cuda-single.csv `
    50
```

## Single PyTorch trials

Activate the validated GPU environment first:

```powershell
.\.venv-gpu\Scripts\Activate.ps1
```

Eager CUDA:

```powershell
python .\benchmarks\bench_pytorch_cuda.py `
    --backend eager `
    --iterations 50 `
    --out .\gpu-results\pytorch-eager-cuda-single.csv
```

Compiled CUDA:

```powershell
python .\benchmarks\bench_pytorch_cuda.py `
    --backend compile `
    --iterations 50 `
    --out .\gpu-results\pytorch-compile-cuda-single.csv
```

## Nine-trial protocol

```powershell
$GpuResults = ".\gpu-results-nine-trial"
Remove-Item $GpuResults -Recurse -Force -ErrorAction SilentlyContinue

python .\benchmarks\run_gpu_benchmark_protocol.py `
    --geo-exe .\build\geo-v8-cuda\Release\bench_v8_cuda.exe `
    --pytorch-script .\benchmarks\bench_pytorch_cuda.py `
    --out-dir $GpuResults `
    --trials 9 `
    --iterations 50
```

For a first smoke run that avoids the longer `torch.compile` phase:

```powershell
python .\benchmarks\run_gpu_benchmark_protocol.py `
    --geo-exe .\build\geo-v8-cuda\Release\bench_v8_cuda.exe `
    --pytorch-script .\benchmarks\bench_pytorch_cuda.py `
    --out-dir .\gpu-results-smoke `
    --trials 3 `
    --iterations 10 `
    --skip-compile
```

## Scientific boundary

Resident, transfer+compute, and end-to-end rows are separate timing classes. Compare only matching precision, workload, side, batch, mode, and timing class. Report geometric means as primary summaries. Peak ratios are descriptive only and cannot replace aggregate results.
