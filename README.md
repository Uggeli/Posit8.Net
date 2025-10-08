# Posit8.Net

Posit8es1 arithmetic library using precomputed lookup tables. Supports CPU and GPU (OpenCL) operations.

## Features

- Arithmetic operations via lookup tables
- 8-bit storage (4x smaller than 32-bit float)
- OpenCL support for GPU matrix operations
- Parallel CPU matrix operations

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

// Arithmetic
byte sum = Posit8Tables.Add(p1, p2);
byte product = Posit8Tables.Mul(p1, p2);

// Decode back to double
double result = Posit8Tables.ToDouble(product);
```

### Vector Operations

```csharp
// Dot product
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

// Single-threaded
Posit8Tables.MatMulDouble(matrixA, matrixB, matrixC, size, size, size);

// Parallel
Posit8Tables.MatMulDoubleParallel(matrixA, matrixB, matrixC, size, size, size);
```

### Matrix Multiplication (GPU)

```csharp
try
{
    using var gpu = new Posit8OpenCL();

    int size = 3072;
    byte[] matrixA = new byte[size * size];
    byte[] matrixB = new byte[size * size];
    byte[] matrixC = new byte[size * size];

    // ... fill matrices ...

    gpu.MatMul(matrixA, matrixB, matrixC, size, size, size);
}
catch (Posit8OpenCLException ex)
{
    Console.WriteLine($"OpenCL not available: {ex.Message}");
}
```

## Performance

- Memory: 4x smaller than 32-bit float
- CPU operations use lookup tables
- GPU operations via OpenCL

Performance varies by hardware, matrix size, and workload.

## Precision

- 8-bit format with limited precision
- Matrix operations accumulate in double precision
- Suitable for: ML inference, approximate computing, compression
- Not suitable for: High-precision scientific computing, financial calculations

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
// CPU single-threaded
void MatMulDouble(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C, int m, int k, int n)

// CPU parallel
void MatMulDoubleParallel(byte[] A, byte[] B, byte[] C, int m, int k, int n)

// GPU via OpenCL
void MatMul(byte[] A, byte[] B, byte[] C, int M, int N, int K)
```

## Requirements

- .NET 9.0
- OpenCL-capable GPU (optional, for GPU operations)
- Compatible GPU drivers with OpenCL support (AMD/NVIDIA/Intel)

## Use Cases

- ML model inference
- Embedded systems
- Approximate computing

## Technical Details

- Format: Posit8es1 (8-bit, exponent size = 1)
- Range: ±4096 with tapered precision
- CPU tables: ~260KB (4 × 64KB tables + utilities)
- GPU memory: 2KB lookup table
- NaR: 0x80 (Not a Real)
- Rounding: Round-to-nearest-ties-to-even with carry propagation
- Matrix operations use double precision accumulation

## License

MIT
