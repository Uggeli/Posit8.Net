using Xunit;

namespace Posit8.Net.Tests;

public class BasicOperationsTests
{
    [Fact]
    public void Neg_PositiveValue_ReturnsNegative()
    {
        byte positive = Posit8Tables.EncodeDouble(5.0);
        byte negative = Posit8Tables.Neg(positive);
        double decoded = Posit8Tables.ToDouble(negative);

        Assert.True(decoded < 0, $"Expected negative, got {decoded}");
        Assert.True(Math.Abs(decoded - (-5.0)) < 0.5, $"Expected ~-5.0, got {decoded}");
    }

    [Fact]
    public void Neg_NegativeValue_ReturnsPositive()
    {
        byte negative = Posit8Tables.EncodeDouble(-5.0);
        byte positive = Posit8Tables.Neg(negative);
        double decoded = Posit8Tables.ToDouble(positive);

        Assert.True(decoded > 0, $"Expected positive, got {decoded}");
        Assert.True(Math.Abs(decoded - 5.0) < 0.5, $"Expected ~5.0, got {decoded}");
    }

    [Fact]
    public void Abs_NegativeValue_ReturnsPositive()
    {
        byte negative = Posit8Tables.EncodeDouble(-5.0);
        byte absolute = Posit8Tables.Abs(negative);
        double decoded = Posit8Tables.ToDouble(absolute);

        Assert.True(decoded > 0, $"Expected positive, got {decoded}");
        Assert.True(Math.Abs(decoded - 5.0) < 0.5, $"Expected ~5.0, got {decoded}");
    }

    [Fact]
    public void Abs_PositiveValue_Unchanged()
    {
        byte positive = Posit8Tables.EncodeDouble(5.0);
        byte absolute = Posit8Tables.Abs(positive);

        Assert.Equal(positive, absolute);
    }

    [Fact]
    public void Recip_Two_ReturnsHalf()
    {
        byte two = Posit8Tables.EncodeDouble(2.0);
        byte recip = Posit8Tables.Recip(two);
        double decoded = Posit8Tables.ToDouble(recip);

        Assert.True(Math.Abs(decoded - 0.5) < 0.05, $"Expected ~0.5, got {decoded}");
    }

    [Fact]
    public void Compare_OrderedValues_ReturnsCorrectComparison()
    {
        byte small = Posit8Tables.EncodeDouble(1.0);
        byte large = Posit8Tables.EncodeDouble(10.0);

        Assert.True(Posit8Tables.Compare(small, large) < 0, "1 < 10");
        Assert.True(Posit8Tables.Compare(large, small) > 0, "10 > 1");
        Assert.Equal(0, Posit8Tables.Compare(small, small));
    }

    [Fact]
    public void Compare_NegativePositive_NegativeIsSmaller()
    {
        byte negative = Posit8Tables.EncodeDouble(-5.0);
        byte positive = Posit8Tables.EncodeDouble(5.0);

        Assert.True(Posit8Tables.Compare(negative, positive) < 0, "-5 < 5");
        Assert.True(Posit8Tables.Compare(positive, negative) > 0, "5 > -5");
    }

    [Fact]
    public void Div_SimpleDivision_ApproximatelyCorrect()
    {
        byte numerator = Posit8Tables.EncodeDouble(10.0);
        byte denominator = Posit8Tables.EncodeDouble(2.0);
        byte result = Posit8Tables.Div(numerator, denominator);
        double decoded = Posit8Tables.ToDouble(result);

        Assert.True(Math.Abs(decoded - 5.0) < 0.5, $"Expected ~5, got {decoded}");
    }

    [Fact]
    public void Div_ByZero_ReturnsNaR()
    {
        byte numerator = Posit8Tables.EncodeDouble(10.0);
        byte zero = Posit8Tables.EncodeDouble(0.0);
        byte result = Posit8Tables.Div(numerator, zero);

        Assert.Equal(0x80, result);
    }

    [Fact]
    public void Sub_SimpleDifference_ApproximatelyCorrect()
    {
        byte a = Posit8Tables.EncodeDouble(10.0);
        byte b = Posit8Tables.EncodeDouble(3.0);
        byte result = Posit8Tables.Sub(a, b);
        double decoded = Posit8Tables.ToDouble(result);

        Assert.True(Math.Abs(decoded - 7.0) < 0.7, $"Expected ~7, got {decoded}");
    }

    [Fact]
    public void AddVector_SimpleVectors_CorrectElementWise()
    {
        byte[] a = {
            Posit8Tables.EncodeDouble(1.0),
            Posit8Tables.EncodeDouble(2.0),
            Posit8Tables.EncodeDouble(3.0)
        };
        byte[] b = {
            Posit8Tables.EncodeDouble(4.0),
            Posit8Tables.EncodeDouble(5.0),
            Posit8Tables.EncodeDouble(6.0)
        };
        byte[] result = new byte[3];

        Posit8Tables.AddVector(a, b, result);

        double[] expected = { 5.0, 7.0, 9.0 };
        for (int i = 0; i < 3; i++)
        {
            double decoded = Posit8Tables.ToDouble(result[i]);
            double relativeError = Math.Abs((decoded - expected[i]) / expected[i]);
            Assert.True(relativeError < 0.15,
                $"Element {i}: expected {expected[i]}, got {decoded}, error {relativeError:P2}");
        }
    }

}
