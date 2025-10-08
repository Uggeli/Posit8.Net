using BenchmarkDotNet.Attributes;
using Posit8.Net;

namespace Posit8.Net.Benchmarks;

[MemoryDiagnoser]
public class DotProductBenchmarks
{
    private byte[] _a = null!;
    private byte[] _b = null!;
    private double[] _aDouble = null!;
    private double[] _bDouble = null!;

    [Params(100, 1000, 10000, 100000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _a = new byte[N];
        _b = new byte[N];
        _aDouble = new double[N];
        _bDouble = new double[N];

        for (int i = 0; i < N; i++)
        {
            _aDouble[i] = (rng.NextDouble() * 2) - 1;
            _bDouble[i] = (rng.NextDouble() * 2) - 1;
            _a[i] = Posit8Tables.EncodeDouble(_aDouble[i]);
            _b[i] = Posit8Tables.EncodeDouble(_bDouble[i]);
        }
    }

    [Benchmark(Baseline = true)]
    public double DotProduct_Float64()
    {
        double sum = 0.0;
        for (int i = 0; i < N; i++)
        {
            sum += _aDouble[i] * _bDouble[i];
        }
        return sum;
    }

    [Benchmark]
    public double DotProduct_Posit8()
    {
        return Posit8Tables.DotProduct(_a, _b);
    }
}
