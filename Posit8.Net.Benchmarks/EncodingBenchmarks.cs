using BenchmarkDotNet.Attributes;
using Posit8.Net;

namespace Posit8.Net.Benchmarks;

[MemoryDiagnoser]
public class EncodingBenchmarks
{
    private double[] _values = null!;
    private byte[] _posits = null!;

    [Params(100, 1000, 10000, 100000)]
    public int N;

    [GlobalSetup]
    public void Setup()
    {
        var rng = new Random(42);
        _values = new double[N];
        _posits = new byte[N];

        for (int i = 0; i < N; i++)
        {
            _values[i] = (rng.NextDouble() * 200) - 100; // Range -100 to 100
            _posits[i] = Posit8Tables.EncodeDouble(_values[i]);
        }
    }

    [Benchmark(Baseline = true)]
    public void EncodeDouble()
    {
        for (int i = 0; i < N; i++)
        {
            _posits[i] = Posit8Tables.EncodeDouble(_values[i]);
        }
    }

    [Benchmark]
    public void DecodeToDouble()
    {
        for (int i = 0; i < N; i++)
        {
            _values[i] = Posit8Tables.ToDouble(_posits[i]);
        }
    }
}
