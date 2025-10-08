using Xunit;

namespace Posit8.Net.Tests;

public class EdgeCaseTests
{
    [Theory]
    [InlineData(0x00)] // Zero
    [InlineData(0x01)] // Smallest positive
    [InlineData(0x7F)] // Largest positive
    [InlineData(0x80)] // NaR
    [InlineData(0x81)] // Largest negative
    [InlineData(0xFF)] // Smallest negative
    public void ToDouble_BoundaryValues_DecodesCorrectly(byte posit)
    {
        double decoded = Posit8Tables.ToDouble(posit);

        if (posit == 0x80)
        {
            Assert.True(double.IsNaN(decoded), "NaR should decode to NaN");
        }
        else if (posit == 0x00)
        {
            Assert.Equal(0.0, decoded);
        }
        else
        {
            Assert.False(double.IsNaN(decoded), $"Non-NaR posit {posit:X2} should not be NaN");
            Assert.False(double.IsInfinity(decoded), $"Posit {posit:X2} should not be infinity");
        }
    }

    [Fact]
    public void Encode_NegativeZero_SameAsPositiveZero()
    {
        byte posZero = Posit8Tables.EncodeDouble(0.0);
        byte negZero = Posit8Tables.EncodeDouble(-0.0);

        Assert.Equal(posZero, negZero);
        Assert.Equal(0, posZero);
    }

    [Fact]
    public void Addition_Commutative()
    {
        Random rng = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            byte a = (byte)rng.Next(256);
            byte b = (byte)rng.Next(256);

            byte ab = Posit8Tables.Add(a, b);
            byte ba = Posit8Tables.Add(b, a);

            Assert.Equal(ab, ba);
        }
    }

    [Fact]
    public void Multiplication_Commutative()
    {
        Random rng = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            byte a = (byte)rng.Next(256);
            byte b = (byte)rng.Next(256);

            byte ab = Posit8Tables.Mul(a, b);
            byte ba = Posit8Tables.Mul(b, a);

            Assert.Equal(ab, ba);
        }
    }

    [Fact]
    public void Addition_IdentityElement()
    {
        byte zero = Posit8Tables.EncodeDouble(0.0);
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            byte result1 = Posit8Tables.Add(value, zero);
            byte result2 = Posit8Tables.Add(zero, value);

            Assert.Equal(value, result1);
            Assert.Equal(value, result2);
        }
    }

    [Fact]
    public void Multiplication_IdentityElement()
    {
        byte one = Posit8Tables.EncodeDouble(1.0);
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            if (value == 0x80) continue; // Skip NaR

            byte result1 = Posit8Tables.Mul(value, one);
            byte result2 = Posit8Tables.Mul(one, value);

            // Allow small rounding differences
            double original = Posit8Tables.ToDouble(value);
            double res1 = Posit8Tables.ToDouble(result1);
            double res2 = Posit8Tables.ToDouble(result2);

            double error1 = Math.Abs(res1 - original);
            double error2 = Math.Abs(res2 - original);

            Assert.True(error1 < 0.1 || error1 / Math.Max(Math.Abs(original), 1) < 0.1,
                $"value={value:X2}, value*1={result1:X2}, error={error1}");
            Assert.True(error2 < 0.1 || error2 / Math.Max(Math.Abs(original), 1) < 0.1,
                $"value={value:X2}, 1*value={result2:X2}, error={error2}");
        }
    }

    [Fact]
    public void Multiplication_ZeroAnnihilator()
    {
        byte zero = Posit8Tables.EncodeDouble(0.0);
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            if (value == 0x80) continue; // NaR * 0 = NaR

            byte result1 = Posit8Tables.Mul(value, zero);
            byte result2 = Posit8Tables.Mul(zero, value);

            Assert.Equal(zero, result1);
            Assert.Equal(zero, result2);
        }
    }

    [Fact]
    public void Compare_ReflexiveProperty()
    {
        // a == a should always be true (compare returns 0)
        for (int i = 0; i < 256; i++)
        {
            byte value = (byte)i;
            int result = Posit8Tables.Compare(value, value);

            Assert.Equal(0, result);
        }
    }

    [Fact]
    public void Compare_AntisymmetricProperty()
    {
        // If Compare(a,b) < 0, then Compare(b,a) > 0
        Random rng = new Random(42);

        for (int i = 0; i < 100; i++)
        {
            byte a = (byte)rng.Next(256);
            byte b = (byte)rng.Next(256);

            int ab = Posit8Tables.Compare(a, b);
            int ba = Posit8Tables.Compare(b, a);

            if (ab < 0)
                Assert.True(ba > 0, $"Compare({a:X2},{b:X2})={ab}, but Compare({b:X2},{a:X2})={ba}");
            else if (ab > 0)
                Assert.True(ba < 0, $"Compare({a:X2},{b:X2})={ab}, but Compare({b:X2},{a:X2})={ba}");
            else
                Assert.Equal(0, ba);
        }
    }

    [Fact]
    public void Subtraction_SelfIsZero()
    {
        byte zero = Posit8Tables.EncodeDouble(0.0);
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            if (value == 0x80) continue; // NaR - NaR = NaR

            byte result = Posit8Tables.Sub(value, value);

            // Should be zero or very close
            double decoded = Posit8Tables.ToDouble(result);
            Assert.True(Math.Abs(decoded) < 0.01,
                $"{value:X2} - {value:X2} = {result:X2} ({decoded}), expected ~0");
        }
    }

    [Fact]
    public void Division_SelfIsOne()
    {
        byte one = Posit8Tables.EncodeDouble(1.0);
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            if (value == 0x80 || value == 0x00) continue; // Skip NaR and zero

            byte result = Posit8Tables.Div(value, value);
            double decoded = Posit8Tables.ToDouble(result);

            // Should be approximately 1
            Assert.True(Math.Abs(decoded - 1.0) < 0.2,
                $"{value:X2} / {value:X2} = {result:X2} ({decoded}), expected ~1");
        }
    }

    [Fact]
    public void Abs_Idempotent()
    {
        // Abs(Abs(x)) == Abs(x)
        Random rng = new Random(42);

        for (int i = 0; i < 50; i++)
        {
            byte value = (byte)rng.Next(256);

            byte abs1 = Posit8Tables.Abs(value);
            byte abs2 = Posit8Tables.Abs(abs1);

            Assert.Equal(abs1, abs2);
        }
    }

    [Fact]
    public void Recip_ZeroAndNaR_ReturnsNaR()
    {
        byte zero = Posit8Tables.EncodeDouble(0.0);
        byte nar = 0x80;

        Assert.Equal(nar, Posit8Tables.Recip(zero));
        Assert.Equal(nar, Posit8Tables.Recip(nar));
    }

    [Fact]
    public void Recip_Involutive()
    {
        // Recip(Recip(x)) should be approximately x
        double[] testValues = { 2.0, 4.0, 0.5, 10.0, 100.0 };

        foreach (var value in testValues)
        {
            byte posit = Posit8Tables.EncodeDouble(value);
            byte recip1 = Posit8Tables.Recip(posit);
            byte recip2 = Posit8Tables.Recip(recip1);

            double original = Posit8Tables.ToDouble(posit);
            double result = Posit8Tables.ToDouble(recip2);

            double error = Math.Abs(result - original) / original;

            Assert.True(error < 0.2,
                $"1/(1/{value}) = {result}, expected ~{value}, error {error:P2}");
        }
    }


    [Fact]
    public void MatMul_1x1Matrices_EqualsScalarMultiplication()
    {
        double a = 7.0;
        double b = 3.0;

        byte[] matA = { Posit8Tables.EncodeDouble(a) };
        byte[] matB = { Posit8Tables.EncodeDouble(b) };
        byte[] matC = new byte[1];

        Posit8Tables.MatMulDouble(matA, matB, matC, 1, 1, 1);

        byte scalarResult = Posit8Tables.Mul(matA[0], matB[0]);

        Assert.Equal(scalarResult, matC[0]);
    }

    [Fact]
    public void DotProduct_SingleElement_EqualsMultiplication()
    {
        byte a = Posit8Tables.EncodeDouble(5.0);
        byte b = Posit8Tables.EncodeDouble(3.0);

        double dotResult = Posit8Tables.DotProduct(new[] { a }, new[] { b });
        double mulResult = Posit8Tables.ToDouble(Posit8Tables.Mul(a, b));

        Assert.True(Math.Abs(dotResult - mulResult) < 0.01,
            $"Dot product single element: {dotResult}, multiplication: {mulResult}");
    }
}
