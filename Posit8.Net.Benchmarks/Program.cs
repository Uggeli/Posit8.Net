using BenchmarkDotNet.Running;

namespace Posit8.Net.Benchmarks;

class Program
{
    static void Main(string[] args)
    {
        // Run all benchmarks or specific ones based on command line args
        if (args.Length == 0)
        {
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
        }
        else
        {
            BenchmarkSwitcher.FromAssembly(typeof(Program).Assembly).Run(args);
        }
    }
}
