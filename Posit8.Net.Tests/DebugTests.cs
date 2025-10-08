using Xunit;
using Xunit.Abstractions;

namespace Posit8.Net.Tests;

public class DebugTests
{
    private readonly ITestOutputHelper _output;

    public DebugTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Debug_CompareEncodings()
    {
        byte neg5 = Posit8Tables.EncodeDouble(-5.0);
        byte pos5 = Posit8Tables.EncodeDouble(5.0);

        _output.WriteLine($"-5.0 encodes to: 0x{neg5:X2} = {neg5} = {Convert.ToString(neg5, 2).PadLeft(8, '0')}");
        _output.WriteLine($"+5.0 encodes to: 0x{pos5:X2} = {pos5} = {Convert.ToString(pos5, 2).PadLeft(8, '0')}");

        double decoded_neg = Posit8Tables.ToDouble(neg5);
        double decoded_pos = Posit8Tables.ToDouble(pos5);

        _output.WriteLine($"Decoded -5: {decoded_neg}");
        _output.WriteLine($"Decoded +5: {decoded_pos}");

        int compareResult = Posit8Tables.Compare(neg5, pos5);
        _output.WriteLine($"Compare(-5, +5) = {compareResult}");

        sbyte sa = (sbyte)(neg5 ^ 0x80);
        sbyte sb = (sbyte)(pos5 ^ 0x80);
        _output.WriteLine($"After XOR: sa={sa}, sb={sb}");
        _output.WriteLine($"sa.CompareTo(sb) = {sa.CompareTo(sb)}");
    }
}
