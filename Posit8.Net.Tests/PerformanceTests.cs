using Xunit;
using Xunit.Abstractions;
using System.Diagnostics;

namespace Posit8.Net.Tests;

public class PerformanceTests
{
    private readonly ITestOutputHelper _output;

    public PerformanceTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Benchmark_DotProduct_CompareWithFloat32()
    {
        int size = 10000;
        Random rng = new Random(42);

        // Generate test data
        float[] floatA = Enumerable.Range(0, size).Select(_ => (float)rng.NextDouble()).ToArray();
        float[] floatB = Enumerable.Range(0, size).Select(_ => (float)rng.NextDouble()).ToArray();
        byte[] posit8A = floatA.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] posit8B = floatB.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();

        // Warm up
        Posit8Tables.DotProduct(posit8A, posit8B);
        DotProductFloat32(floatA, floatB);

        // Benchmark Float32
        Stopwatch sw = Stopwatch.StartNew();
        double float32Result = DotProductFloat32(floatA, floatB);
        sw.Stop();
        long float32Time = sw.ElapsedMilliseconds;

        // Benchmark Posit8
        sw.Restart();
        double posit8Result = Posit8Tables.DotProduct(posit8A, posit8B);
        sw.Stop();
        long posit8Time = sw.ElapsedMilliseconds;

        _output.WriteLine($"DotProduct ({size} elements):");
        _output.WriteLine($"  Float32: {float32Time}ms, Result: {float32Result}");
        _output.WriteLine($"  Posit8:  {posit8Time}ms, Result: {posit8Result}");
        _output.WriteLine($"  Error: {Math.Abs((posit8Result - float32Result) / float32Result):P2}");

        // Just report, don't fail on performance
        Assert.True(true);
    }

    [Theory]
    [InlineData(64)]
    [InlineData(128)]
    [InlineData(256)]
    public void Benchmark_MatMul_CompareWithFloat32(int size)
    {
        Random rng = new Random(42);

        // Generate test data
        float[] floatA = Enumerable.Range(0, size * size).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        float[] floatB = Enumerable.Range(0, size * size).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        byte[] posit8A = floatA.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] posit8B = floatB.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();

        // Warm up
        byte[] posit8C = new byte[size * size];
        Posit8Tables.MatMulDouble(posit8A, posit8B, posit8C, size, size, size);

        float[] float32C = new float[size * size];
        MatMulFloat32(floatA, floatB, float32C, size);

        // Benchmark Float32
        Stopwatch sw = Stopwatch.StartNew();
        MatMulFloat32(floatA, floatB, float32C, size);
        sw.Stop();
        long float32Time = sw.ElapsedMilliseconds;

        // Benchmark Posit8
        sw.Restart();
        Posit8Tables.MatMulDouble(posit8A, posit8B, posit8C, size, size, size);
        sw.Stop();
        long posit8Time = sw.ElapsedMilliseconds;

        // Calculate error
        double maxError = 0;
        for (int i = 0; i < size * size; i++)
        {
            double expected = float32C[i];
            double actual = Posit8Tables.ToDouble(posit8C[i]);
            if (Math.Abs(expected) > 0.01)
            {
                double error = Math.Abs((actual - expected) / expected);
                maxError = Math.Max(maxError, error);
            }
        }

        _output.WriteLine($"MatMul ({size}x{size}):");
        _output.WriteLine($"  Float32: {float32Time}ms");
        _output.WriteLine($"  Posit8:  {posit8Time}ms");
        _output.WriteLine($"  Memory: Float32={size * size * 4 / 1024}KB, Posit8={size * size / 1024}KB");
        _output.WriteLine($"  Max error: {maxError:P2}");

        Assert.True(true);
    }

    [Fact]
    public void Benchmark_MemoryFootprint()
    {
        int[] sizes = { 512, 1024, 2048, 4096 };

        _output.WriteLine("Memory footprint comparison:");
        _output.WriteLine("Size\tFloat32\t\tPosit8\t\tSavings");

        foreach (int size in sizes)
        {
            long float32Bytes = (long)size * size * 4;
            long posit8Bytes = (long)size * size * 1;
            double savingsRatio = (double)float32Bytes / posit8Bytes;

            _output.WriteLine($"{size}x{size}\t{float32Bytes / 1024 / 1024}MB\t\t{posit8Bytes / 1024 / 1024}MB\t\t{savingsRatio:F1}x");
        }

        Assert.True(true);
    }

    [Fact]
    public void Benchmark_EncodingOverhead()
    {
        int count = 100000;
        Random rng = new Random(42);
        double[] values = Enumerable.Range(0, count).Select(_ => rng.NextDouble() * 100).ToArray();

        // Warm up
        for (int i = 0; i < 1000; i++)
        {
            Posit8Tables.EncodeDouble(values[i]);
        }

        // Benchmark encoding
        Stopwatch sw = Stopwatch.StartNew();
        byte[] encoded = new byte[count];
        for (int i = 0; i < count; i++)
        {
            encoded[i] = Posit8Tables.EncodeDouble(values[i]);
        }
        sw.Stop();
        long encodeTime = sw.ElapsedMilliseconds;

        // Benchmark decoding
        sw.Restart();
        double[] decoded = new double[count];
        for (int i = 0; i < count; i++)
        {
            decoded[i] = Posit8Tables.ToDouble(encoded[i]);
        }
        sw.Stop();
        long decodeTime = sw.ElapsedMilliseconds;

        _output.WriteLine($"Encoding/Decoding ({count} values):");
        _output.WriteLine($"  Encode: {encodeTime}ms ({count / (double)encodeTime * 1000:F0} ops/sec)");
        _output.WriteLine($"  Decode: {decodeTime}ms ({count / (double)decodeTime * 1000:F0} ops/sec)");

        Assert.True(true);
    }

    // Helper methods
    private static double DotProductFloat32(float[] a, float[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            sum += a[i] * b[i];
        }
        return sum;
    }

    private static void MatMulFloat32(float[] A, float[] B, float[] C, int size)
    {
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                float sum = 0;
                for (int k = 0; k < size; k++)
                {
                    sum += A[row * size + k] * B[k * size + col];
                }
                C[row * size + col] = sum;
            }
        }
    }
}
