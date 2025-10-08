using Posit8.Net;
using System.Diagnostics;

Console.WriteLine("=== Posit8.Net Samples ===\n");

// Sample 1: Basic Encode/Decode
Console.WriteLine("1. Basic Encode/Decode");
Console.WriteLine("----------------------");
double[] values = { 0.0, 1.0, -1.0, 3.14, 100.0, -50.0, 0.001 };
foreach (var value in values)
{
    byte encoded = Posit8Tables.EncodeDouble(value);
    double decoded = Posit8Tables.ToDouble(encoded);
    double error = Math.Abs(decoded - value);
    Console.WriteLine($"  {value,8:F3} -> 0x{encoded:X2} -> {decoded,8:F3} (error: {error:F6})");
}
Console.WriteLine();

// Sample 2: Basic Arithmetic
Console.WriteLine("2. Basic Arithmetic");
Console.WriteLine("-------------------");
double a = 5.0, b = 3.0;
byte pa = Posit8Tables.EncodeDouble(a);
byte pb = Posit8Tables.EncodeDouble(b);

Console.WriteLine($"  a = {a}, b = {b}");
Console.WriteLine($"  a + b = {Posit8Tables.ToDouble(Posit8Tables.Add(pa, pb)):F3} (expected: {a + b})");
Console.WriteLine($"  a - b = {Posit8Tables.ToDouble(Posit8Tables.Sub(pa, pb)):F3} (expected: {a - b})");
Console.WriteLine($"  a * b = {Posit8Tables.ToDouble(Posit8Tables.Mul(pa, pb)):F3} (expected: {a * b})");
Console.WriteLine($"  a / b = {Posit8Tables.ToDouble(Posit8Tables.Div(pa, pb)):F3} (expected: {a / b:F3})");
Console.WriteLine();

// Sample 3: Vector Dot Product
Console.WriteLine("3. Vector Dot Product");
Console.WriteLine("---------------------");
int vecSize = 1000;
var rng = new Random(42);
double[] vec1 = Enumerable.Range(0, vecSize).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
double[] vec2 = Enumerable.Range(0, vecSize).Select(_ => rng.NextDouble() * 2 - 1).ToArray();

// Convert to Posit8
byte[] pvec1 = vec1.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
byte[] pvec2 = vec2.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();

// Compute dot products
double dotFloat64 = vec1.Zip(vec2, (x, y) => x * y).Sum();
double dotPosit8 = Posit8Tables.DotProduct(pvec1, pvec2);

Console.WriteLine($"  Vector size: {vecSize}");
Console.WriteLine($"  Float64 result: {dotFloat64:F6}");
Console.WriteLine($"  Posit8 result:  {dotPosit8:F6}");
Console.WriteLine($"  Error: {Math.Abs(dotFloat64 - dotPosit8):F6} ({Math.Abs((dotPosit8 - dotFloat64) / dotFloat64):P2})");
Console.WriteLine();

// Sample 4: Matrix Multiplication (CPU)
Console.WriteLine("4. Matrix Multiplication (CPU)");
Console.WriteLine("-------------------------------");
int matSize = 128;
byte[] matA = new byte[matSize * matSize];
byte[] matB = new byte[matSize * matSize];
byte[] matC = new byte[matSize * matSize];

// Fill with random values
for (int i = 0; i < matA.Length; i++)
{
    matA[i] = Posit8Tables.EncodeDouble(rng.NextDouble() * 2 - 1);
    matB[i] = Posit8Tables.EncodeDouble(rng.NextDouble() * 2 - 1);
}

// Single-threaded
var sw = Stopwatch.StartNew();
Posit8Tables.MatMulDouble(matA, matB, matC, matSize, matSize, matSize);
sw.Stop();
Console.WriteLine($"  Single-threaded ({matSize}x{matSize}): {sw.ElapsedMilliseconds}ms");

// Parallel
sw.Restart();
Posit8Tables.MatMulDoubleParallel(matA, matB, matC, matSize, matSize, matSize);
sw.Stop();
Console.WriteLine($"  Parallel ({matSize}x{matSize}):        {sw.ElapsedMilliseconds}ms");
Console.WriteLine();

// Sample 5: GPU Matrix Multiplication (if available)
Console.WriteLine("5. Matrix Multiplication (GPU)");
Console.WriteLine("-------------------------------");
try
{
    using var gpu = new Posit8OpenCL();
    Console.WriteLine("  OpenCL initialized successfully");

    int gpuSize = 256;
    byte[] gpuA = new byte[gpuSize * gpuSize];
    byte[] gpuB = new byte[gpuSize * gpuSize];
    byte[] gpuC = new byte[gpuSize * gpuSize];

    for (int i = 0; i < gpuA.Length; i++)
    {
        gpuA[i] = Posit8Tables.EncodeDouble(rng.NextDouble() * 2 - 1);
        gpuB[i] = Posit8Tables.EncodeDouble(rng.NextDouble() * 2 - 1);
    }

    sw.Restart();
    gpu.MatMul(gpuA, gpuB, gpuC, gpuSize, gpuSize, gpuSize);
    sw.Stop();

    Console.WriteLine($"  GPU ({gpuSize}x{gpuSize}): {sw.ElapsedMilliseconds}ms");
}
catch (Posit8OpenCLException ex)
{
    Console.WriteLine($"  GPU not available: {ex.Message}");
    Console.WriteLine("  (This is expected if you don't have OpenCL installed)");
}
Console.WriteLine();

// Sample 6: Simple 2-Layer Neural Network Forward Pass
Console.WriteLine("6. Simple 2-Layer Neural Network");
Console.WriteLine("----------------------------------");
int inputSize = 64;
int hiddenSize = 32;
int outputSize = 10;

// Create random weights and bias
byte[] w1 = new byte[inputSize * hiddenSize];
byte[] b1 = new byte[hiddenSize];
byte[] w2 = new byte[hiddenSize * outputSize];
byte[] b2 = new byte[outputSize];

for (int i = 0; i < w1.Length; i++)
    w1[i] = Posit8Tables.EncodeDouble((rng.NextDouble() * 2 - 1) * 0.1);
for (int i = 0; i < w2.Length; i++)
    w2[i] = Posit8Tables.EncodeDouble((rng.NextDouble() * 2 - 1) * 0.1);
for (int i = 0; i < b1.Length; i++)
    b1[i] = Posit8Tables.EncodeDouble(0.0);
for (int i = 0; i < b2.Length; i++)
    b2[i] = Posit8Tables.EncodeDouble(0.0);

// Create input
byte[] input = new byte[inputSize];
for (int i = 0; i < inputSize; i++)
    input[i] = Posit8Tables.EncodeDouble(rng.NextDouble());

// Forward pass: hidden = input @ w1 + b1
byte[] hidden = new byte[hiddenSize];
Posit8Tables.MatMulDouble(input, w1, hidden, 1, inputSize, hiddenSize);
for (int i = 0; i < hiddenSize; i++)
    hidden[i] = Posit8Tables.Add(hidden[i], b1[i]);

// Apply ReLU (simple approximation: max(0, x))
for (int i = 0; i < hiddenSize; i++)
{
    double val = Posit8Tables.ToDouble(hidden[i]);
    if (val < 0) hidden[i] = Posit8Tables.EncodeDouble(0.0);
}

// Forward pass: output = hidden @ w2 + b2
byte[] output = new byte[outputSize];
Posit8Tables.MatMulDouble(hidden, w2, output, 1, hiddenSize, outputSize);
for (int i = 0; i < outputSize; i++)
    output[i] = Posit8Tables.Add(output[i], b2[i]);

Console.WriteLine($"  Network: {inputSize} -> {hiddenSize} -> {outputSize}");
Console.WriteLine($"  Output values:");
for (int i = 0; i < outputSize; i++)
{
    Console.WriteLine($"    output[{i}] = {Posit8Tables.ToDouble(output[i]):F6}");
}
Console.WriteLine();

Console.WriteLine("=== All samples complete! ===");
