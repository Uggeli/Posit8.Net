using BenchmarkDotNet.Attributes;
using Posit8.Net;

namespace Posit8.Net.Benchmarks;

[MemoryDiagnoser]
public class MatMulBenchmarks
{
    private byte[] _a = null!;
    private byte[] _b = null!;
    private byte[] _c = null!;
    private double[] _aDouble = null!;
    private double[] _bDouble = null!;
    private double[] _cDouble = null!;

    [Params(64, 128, 256, 512)]
    public int Size;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        int totalElements = Size * Size;

        _a = new byte[totalElements];
        _b = new byte[totalElements];
        _c = new byte[totalElements];
        _aDouble = new double[totalElements];
        _bDouble = new double[totalElements];
        _cDouble = new double[totalElements];

        for (int i = 0; i < totalElements; i++)
        {
            _aDouble[i] = (rng.NextDouble() * 2) - 1;
            _bDouble[i] = (rng.NextDouble() * 2) - 1;
            _a[i] = Posit8Tables.EncodeDouble(_aDouble[i]);
            _b[i] = Posit8Tables.EncodeDouble(_bDouble[i]);
        }
    }

    [Benchmark(Baseline = true)]
    public void MatMul_Float64()
    {
        for (int row = 0; row < Size; row++)
        {
            for (int col = 0; col < Size; col++)
            {
                double sum = 0;
                for (int k = 0; k < Size; k++)
                {
                    sum += _aDouble[row * Size + k] * _bDouble[k * Size + col];
                }
                _cDouble[row * Size + col] = sum;
            }
        }
    }

    [Benchmark]
    public void MatMul_Posit8()
    {
        Posit8Tables.MatMulDouble(_a, _b, _c, Size, Size, Size);
    }

    [Benchmark]
    public void MatMul_Posit8_Parallel()
    {
        Posit8Tables.MatMulDoubleParallel(_a, _b, _c, Size, Size, Size);
    }
}
