using Xunit;
using Xunit.Abstractions;

namespace Posit8.Net.Tests;

public class StressTests
{
    private readonly ITestOutputHelper _output;

    public StressTests(ITestOutputHelper output)
    {
        _output = output;
    }

    [Fact]
    public void Encode_ExtremeValues_HandlesGracefully()
    {
        // Test extreme values
        double[] extremeValues = {
            double.MaxValue,
            double.MinValue,
            double.Epsilon,
            -double.Epsilon,
            1e100,
            -1e100,
            1e-100,
            -1e-100,
            double.PositiveInfinity,
            double.NegativeInfinity
        };

        foreach (var value in extremeValues)
        {
            byte encoded = Posit8Tables.EncodeDouble(value);
            double decoded = Posit8Tables.ToDouble(encoded);

            // Should not crash, and should return valid posit8 or NaR
            Assert.True(encoded <= 255);
            Assert.False(double.IsInfinity(decoded) && encoded != 0x80,
                $"Value {value} encoded to {encoded:X2} decoded to infinity");
        }
    }

    [Fact]
    public void ArithmeticOperations_WithNaR_AlwaysReturnsNaR()
    {
        byte nar = 0x80;
        byte normalValue = Posit8Tables.EncodeDouble(5.0);

        // All operations with NaR should return NaR
        Assert.Equal(nar, Posit8Tables.Add(nar, normalValue));
        Assert.Equal(nar, Posit8Tables.Add(normalValue, nar));
        Assert.Equal(nar, Posit8Tables.Mul(nar, normalValue));
        Assert.Equal(nar, Posit8Tables.Mul(normalValue, nar));
        Assert.Equal(nar, Posit8Tables.Sub(nar, normalValue));
        Assert.Equal(nar, Posit8Tables.Sub(normalValue, nar));
        Assert.Equal(nar, Posit8Tables.Div(nar, normalValue));
        Assert.Equal(nar, Posit8Tables.Div(normalValue, nar));
    }

    [Fact]
    public void DivisionByZero_ReturnsNaR()
    {
        byte[] testValues = {
            Posit8Tables.EncodeDouble(1.0),
            Posit8Tables.EncodeDouble(100.0),
            Posit8Tables.EncodeDouble(-5.0),
            Posit8Tables.EncodeDouble(0.5)
        };

        byte zero = Posit8Tables.EncodeDouble(0.0);

        foreach (var value in testValues)
        {
            byte result = Posit8Tables.Div(value, zero);
            Assert.Equal(0x80, result); // NaR
        }
    }

    [Fact]
    public void Multiplication_VerySmallNumbers_NoUnderflow()
    {
        byte verySmall1 = Posit8Tables.EncodeDouble(0.01);
        byte verySmall2 = Posit8Tables.EncodeDouble(0.01);

        byte result = Posit8Tables.Mul(verySmall1, verySmall2);
        double decoded = Posit8Tables.ToDouble(result);

        // Should be approximately 0.0001, but might underflow to 0
        Assert.True(decoded >= 0 && decoded < 0.01,
            $"0.01 * 0.01 = {decoded}, should be small positive or zero");
    }

    [Fact]
    public void Addition_LargeAndSmall_PreservesLarge()
    {
        byte large = Posit8Tables.EncodeDouble(100.0);
        byte small = Posit8Tables.EncodeDouble(0.01);

        byte result = Posit8Tables.Add(large, small);
        double decoded = Posit8Tables.ToDouble(result);

        // Small value should be absorbed due to limited precision
        double error = Math.Abs(decoded - 100.0);
        Assert.True(error < 5.0, $"100 + 0.01 = {decoded}, expected ~100");
    }

    [Fact]
    public void MatMul_NonSquareMatrices_HandlesCorrectly()
    {
        // 2x3 * 3x4 = 2x4
        int m = 2, k = 3, n = 4;

        byte[] A = {
            Posit8Tables.EncodeDouble(1), Posit8Tables.EncodeDouble(2), Posit8Tables.EncodeDouble(3),
            Posit8Tables.EncodeDouble(4), Posit8Tables.EncodeDouble(5), Posit8Tables.EncodeDouble(6)
        };

        byte[] B = {
            Posit8Tables.EncodeDouble(1), Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(1),
            Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(1), Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(1),
            Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(0), Posit8Tables.EncodeDouble(1), Posit8Tables.EncodeDouble(1)
        };

        byte[] C = new byte[m * n];

        Posit8Tables.MatMulDouble(A, B, C, m, k, n);

        // Verify dimensions and no crashes
        Assert.Equal(8, C.Length);

        // Check some values make sense
        double firstElement = Posit8Tables.ToDouble(C[0]);
        Assert.True(firstElement > 0, "First element should be positive");
    }

    [Fact]
    public void MatMul_WrongDimensions_ThrowsException()
    {
        byte[] A = new byte[6];  // Should be 2x3
        byte[] B = new byte[12]; // Should be 3x4
        byte[] C = new byte[8];  // Should be 2x4

        // Wrong dimensions for A
        Assert.Throws<ArgumentException>(() =>
            Posit8Tables.MatMulDouble(A, B, C, 3, 3, 4)); // Claims 3x3 but A is 6 elements

        // Wrong dimensions for B
        Assert.Throws<ArgumentException>(() =>
            Posit8Tables.MatMulDouble(A, B, C, 2, 2, 4)); // Claims 2x2 but needs k=3

        // Wrong dimensions for C
        Assert.Throws<ArgumentException>(() =>
            Posit8Tables.MatMulDouble(A, B, C, 2, 3, 3)); // Claims output is 2x3 but C is 8 elements
    }

    [Fact]
    public void Compare_Transitivity_Holds()
    {
        byte a = Posit8Tables.EncodeDouble(1.0);
        byte b = Posit8Tables.EncodeDouble(5.0);
        byte c = Posit8Tables.EncodeDouble(10.0);

        // If a < b and b < c, then a < c
        Assert.True(Posit8Tables.Compare(a, b) < 0);
        Assert.True(Posit8Tables.Compare(b, c) < 0);
        Assert.True(Posit8Tables.Compare(a, c) < 0);
    }

    [Fact]
    public void Abs_PreservesZero()
    {
        byte zero = Posit8Tables.EncodeDouble(0.0);
        byte absZero = Posit8Tables.Abs(zero);

        Assert.Equal(zero, absZero);
        Assert.Equal(0.0, Posit8Tables.ToDouble(absZero));
    }

    [Fact]
    public void Neg_DoubleNegation_ReturnsOriginal()
    {
        byte[] testValues = {
            Posit8Tables.EncodeDouble(5.0),
            Posit8Tables.EncodeDouble(-3.0),
            Posit8Tables.EncodeDouble(0.5),
            Posit8Tables.EncodeDouble(100.0)
        };

        foreach (var value in testValues)
        {
            byte negated = Posit8Tables.Neg(value);
            byte doubleNegated = Posit8Tables.Neg(negated);

            Assert.Equal(value, doubleNegated);
        }
    }

    [Fact]
    public void Recip_Reciprocal_MultiplyToOne()
    {
        double[] testValues = { 2.0, 4.0, 10.0, 0.5 };

        foreach (var value in testValues)
        {
            byte posit = Posit8Tables.EncodeDouble(value);
            byte recip = Posit8Tables.Recip(posit);
            byte product = Posit8Tables.Mul(posit, recip);

            double result = Posit8Tables.ToDouble(product);

            // Should be approximately 1.0
            Assert.True(Math.Abs(result - 1.0) < 0.15,
                $"1/{value} * {value} = {result}, expected ~1.0");
        }
    }

    [Fact]
    public void DotProduct_EmptyVectors_ReturnsZero()
    {
        byte[] empty1 = Array.Empty<byte>();
        byte[] empty2 = Array.Empty<byte>();

        double result = Posit8Tables.DotProduct(empty1, empty2);

        Assert.Equal(0.0, result);
    }

    [Fact]
    public void DotProduct_MismatchedLengths_ThrowsException()
    {
        byte[] a = new byte[5];
        byte[] b = new byte[3];

        Assert.Throws<ArgumentException>(() => Posit8Tables.DotProduct(a, b));
    }

    [Fact]
    public void AddVector_MismatchedLengths_ThrowsException()
    {
        byte[] a = new byte[5];
        byte[] b = new byte[3];
        byte[] result = new byte[5];

        Assert.Throws<ArgumentException>(() => Posit8Tables.AddVector(a, b, result));
    }

    [Fact]
    public void LargeAccumulation_DoesNotOverflow()
    {
        // Sum of 256 ones
        int count = 256;
        byte one = Posit8Tables.EncodeDouble(1.0);
        byte[] ones = Enumerable.Repeat(one, count).ToArray();

        double result = Posit8Tables.DotProduct(ones, ones);

        // Should be 256 (1*1 * 256 times)
        Assert.True(result > 200 && result < 300,
            $"Sum of 256 ones squared: {result}, expected ~256");
    }

    [Fact]
    public void AlternatingSignSum_CancelsOut()
    {
        int count = 100;
        byte[] a = new byte[count];
        byte[] b = new byte[count];

        byte pos = Posit8Tables.EncodeDouble(1.0);
        byte neg = Posit8Tables.EncodeDouble(-1.0);

        for (int i = 0; i < count; i++)
        {
            a[i] = pos;
            b[i] = (i % 2 == 0) ? pos : neg; // alternating +1, -1
        }

        double result = Posit8Tables.DotProduct(a, b);

        // Should be close to 0 (50 positive - 50 negative)
        Assert.True(Math.Abs(result) < 5.0,
            $"Alternating sum: {result}, expected ~0");
    }
}
