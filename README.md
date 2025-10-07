# Posit8.Net

Posit8es1 arithmetic library using precomputed lookup tables. Supports CPU and GPU (OpenCL) operations with 4x memory savings vs Float32.

## Features

- **O(1) Operations**: All arithmetic ops (add, mul, sub, div) via lookup tables
- **Memory Efficient**: 8-bit posit vs 32-bit float = 4x smaller models
- **GPU Accelerated**: OpenCL support for parallel matrix operations
- **ML Ready**: Suitable for inference where memory bandwidth matters

## Installation

```bash
dotnet add package Posit8.Net
```

## Quick Start

### Basic Operations

```csharp
using Posit8.Net;

// Encode double to posit8
byte p1 = Posit8Tables.EncodeDouble(3.5);
byte p2 = Posit8Tables.EncodeDouble(2.0);

// Arithmetic (O(1) table lookups)
byte sum = Posit8Tables.Add(p1, p2);
byte product = Posit8Tables.Mul(p1, p2);

// Decode back to double
double result = Posit8Tables.ToDouble(product);  // 7.0
```

### Vector Operations

```csharp
// Dot product (double accumulation)
byte[] vecA = new byte[1000];
byte[] vecB = new byte[1000];

// ... fill with encoded values ...

double dotProduct = Posit8Tables.DotProduct(vecA, vecB);
```

### Matrix Multiplication (CPU)

```csharp
int size = 512;
byte[] matrixA = new byte[size * size];
byte[] matrixB = new byte[size * size];
byte[] matrixC = new byte[size * size];

// ... fill matrices ...

// CPU: Double accumulation (recommended for accuracy)
Posit8Tables.MatMulDouble(matrixA, matrixB, matrixC, size, size, size);
```

### Matrix Multiplication (GPU)

```csharp
// Initialize OpenCL
if (Posit8OpenCL.Initialize())
{
    int size = 3072;
    byte[] matrixA = new byte[size * size];
    byte[] matrixB = new byte[size * size];
    byte[] matrixC = new byte[size * size];

    // ... fill matrices ...

    // GPU acceleration
    Posit8OpenCL.MatMulGPU(matrixA, matrixB, matrixC,
                           size, size, size,
                           useDoubleAccum: true);

    // Cleanup
    Posit8OpenCL.Cleanup();
}
```

## Performance

### Characteristics

- **Memory**: 4x smaller than Float32 (8-bit vs 32-bit)
- **CPU**: Lookup table operations, performance depends on cache efficiency
- **GPU**: OpenCL acceleration for matrix operations on compatible hardware
- **Trade-offs**: Lower precision (8-bit) for reduced memory bandwidth

Performance will vary based on hardware, matrix size, and workload. Benchmark on your target platform for specific use cases.

## Precision

- **Format**: 8-bit posit with limited precision compared to float32
- **Accumulation**: Errors can compound in large matrix operations
- **Suitable for**: ML inference, approximate computing, compression
- **Not suitable for**: High-precision scientific computing, financial calculations

Posit8 trades precision for memory efficiency. Test accuracy on your specific workload before production use.

## API Reference

### Encoding/Decoding

```csharp
byte EncodeDouble(double value)
double ToDouble(byte posit)
```

### Arithmetic Operations

```csharp
byte Add(byte a, byte b)
byte Sub(byte a, byte b)
byte Mul(byte a, byte b)
byte Div(byte a, byte b)
byte Neg(byte posit)
byte Abs(byte posit)
byte Recip(byte posit)
int Compare(byte a, byte b)
```

### Vector Operations

```csharp
void AddVector(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> result)
double DotProduct(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
```

### Matrix Operations

```csharp
// CPU
void MatMul(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C, int m, int k, int n)
void MatMulDouble(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C, int m, int k, int n)

// GPU
void MatMulGPU(byte[] A, byte[] B, byte[] C, int M, int N, int K, bool useDoubleAccum)
```

## Requirements

- .NET 9.0
- OpenCL-capable GPU (optional, for GPU operations)
- Compatible GPU drivers with OpenCL support (AMD/NVIDIA/Intel)

## Use Cases

- **ML Model Inference**: Compress models by 4x, reduce memory bandwidth
- **Embedded Systems**: Reduced memory footprint
- **Edge AI**: Deploy larger models on memory-constrained devices
- **Approximate Computing**: Where memory efficiency matters more than precision

## Technical Details

- **Format**: Posit8es1 (8-bit, exponent size = 1)
- **Range**: ±4096 with tapered precision
- **Tables**: ~130KB (fits in L2 cache)
- **NaR**: 0x80 (Not a Real - like NaN)

## License

MIT

## Author

Your Name

## Contributing

Contributions welcome! Please open an issue or PR.
