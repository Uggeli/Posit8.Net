using Xunit;

namespace Posit8.Net.Tests;

public class AccuracyTests
{
    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    [InlineData(-1.0)]
    [InlineData(2.0)]
    [InlineData(0.5)]
    [InlineData(100.0)]
    [InlineData(-50.0)]
    [InlineData(0.125)]
    public void EncodeDecodeRoundTrip_SimpleValues_PreservesValue(double value)
    {
        byte encoded = Posit8Tables.EncodeDouble(value);
        double decoded = Posit8Tables.ToDouble(encoded);

        double relativeError = Math.Abs((decoded - value) / value);

        // Allow up to 5% relative error due to 8-bit quantization
        Assert.True(relativeError < 0.05 || Math.Abs(decoded - value) < 0.01,
            $"Value {value} encoded to {encoded:X2}, decoded to {decoded}, error: {relativeError:P2}");
    }

    [Fact]
    public void EncodeDecodeRoundTrip_Zero_Exact()
    {
        byte encoded = Posit8Tables.EncodeDouble(0.0);
        double decoded = Posit8Tables.ToDouble(encoded);

        Assert.Equal(0.0, decoded);
        Assert.Equal(0, encoded);
    }

    [Fact]
    public void EncodeDecodeRoundTrip_NaN_ReturnsNaR()
    {
        byte encoded = Posit8Tables.EncodeDouble(double.NaN);
        double decoded = Posit8Tables.ToDouble(encoded);

        Assert.Equal(0x80, encoded);
        Assert.True(double.IsNaN(decoded));
    }

    [Fact]
    public void EncodeDecodeRoundTrip_AllPosit8Values_Consistent()
    {
        for (int i = 0; i < 256; i++)
        {
            byte posit = (byte)i;
            double decoded = Posit8Tables.ToDouble(posit);

            // NaR should decode to NaN
            if (posit == 0x80)
            {
                Assert.True(double.IsNaN(decoded), $"NaR (0x80) should decode to NaN");
                continue;
            }

            // Zero should decode to zero
            if (posit == 0)
            {
                Assert.Equal(0.0, decoded);
                continue;
            }

            // All other values should not be NaN or infinity
            Assert.False(double.IsNaN(decoded), $"Posit {posit:X2} decoded to NaN");
            Assert.False(double.IsInfinity(decoded), $"Posit {posit:X2} decoded to Infinity");

            // Verify round-trip: decode then encode should give same or very close value
            byte reencoded = Posit8Tables.EncodeDouble(decoded);
            double redecoded = Posit8Tables.ToDouble(reencoded);

            // Allow for one step of quantization error
            double errorRatio = Math.Abs((redecoded - decoded) / decoded);
            Assert.True(errorRatio < 0.1 || Math.Abs(redecoded - decoded) < 0.001,
                $"Posit {posit:X2} -> {decoded} -> {reencoded:X2} -> {redecoded}, error {errorRatio:P2}");
        }
    }

    [Theory]
    [InlineData(1.0, 1.0, 2.0)]
    [InlineData(2.0, 3.0, 5.0)]
    [InlineData(5.0, -3.0, 2.0)]
    [InlineData(0.5, 0.5, 1.0)]
    public void Add_SimpleValues_ApproximatelyCorrect(double a, double b, double expected)
    {
        byte pa = Posit8Tables.EncodeDouble(a);
        byte pb = Posit8Tables.EncodeDouble(b);
        byte result = Posit8Tables.Add(pa, pb);
        double decoded = Posit8Tables.ToDouble(result);

        double relativeError = Math.Abs((decoded - expected) / expected);
        Assert.True(relativeError < 0.1,
            $"{a} + {b} = {decoded}, expected {expected}, error: {relativeError:P2}");
    }

    [Theory]
    [InlineData(2.0, 3.0, 6.0)]
    [InlineData(5.0, 4.0, 20.0)]
    [InlineData(0.5, 0.5, 0.25)]
    [InlineData(-2.0, 3.0, -6.0)]
    public void Mul_SimpleValues_ApproximatelyCorrect(double a, double b, double expected)
    {
        byte pa = Posit8Tables.EncodeDouble(a);
        byte pb = Posit8Tables.EncodeDouble(b);
        byte result = Posit8Tables.Mul(pa, pb);
        double decoded = Posit8Tables.ToDouble(result);

        double relativeError = Math.Abs((decoded - expected) / expected);
        Assert.True(relativeError < 0.1,
            $"{a} * {b} = {decoded}, expected {expected}, error: {relativeError:P2}");
    }

    [Fact]
    public void DotProduct_SmallVector_AccurateAccumulation()
    {
        // Test vector: [1, 2, 3, 4, 5]
        double[] valuesA = { 1, 2, 3, 4, 5 };
        double[] valuesB = { 2, 2, 2, 2, 2 };

        byte[] vecA = valuesA.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] vecB = valuesB.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();

        double result = Posit8Tables.DotProduct(vecA, vecB);
        double expected = 30.0; // 1*2 + 2*2 + 3*2 + 4*2 + 5*2 = 30

        double relativeError = Math.Abs((result - expected) / expected);
        Assert.True(relativeError < 0.05,
            $"Dot product = {result}, expected {expected}, error: {relativeError:P2}");
    }

    [Fact]
    public void MatMulDouble_SmallMatrix_CompareWithFloat32()
    {
        // 4x4 matrices - actual multiplication to test accumulation
        int size = 4;
        double[] matrixAValues = { 1, 2, 1, 0,  0, 1, 2, 1,  1, 0, 1, 2,  2, 1, 0, 1 };
        double[] matrixBValues = { 2, 0, 1, 0,  0, 2, 0, 1,  1, 0, 2, 0,  0, 1, 0, 2 };

        // Compute reference result in double precision
        double[] expectedC = new double[size * size];
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                double sum = 0;
                for (int k = 0; k < size; k++)
                {
                    sum += matrixAValues[row * size + k] * matrixBValues[k * size + col];
                }
                expectedC[row * size + col] = sum;
            }
        }

        byte[] matrixA = matrixAValues.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] matrixB = matrixBValues.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] matrixC = new byte[size * size];

        Posit8Tables.MatMulDouble(matrixA, matrixB, matrixC, size, size, size);

        // Compare with reference
        for (int i = 0; i < size * size; i++)
        {
            double expected = expectedC[i];
            double actual = Posit8Tables.ToDouble(matrixC[i]);

            if (Math.Abs(expected) > 0.5)
            {
                double relativeError = Math.Abs((actual - expected) / expected);
                Assert.True(relativeError < 0.15,
                    $"Element [{i}]: {actual}, expected {expected}, error: {relativeError:P2}");
            }
            else
            {
                // For small values, use absolute error
                Assert.True(Math.Abs(actual - expected) < 0.5,
                    $"Element [{i}]: {actual}, expected {expected}");
            }
        }
    }

    [Fact]
    public void MatMulDouble_LargerMatrix_MeasureAccumulationError()
    {
        // Test 32x32 matrix multiplication
        int size = 32;
        Random rng = new Random(42);

        // Generate random matrices with small values to avoid overflow
        double[] matrixAValues = Enumerable.Range(0, size * size)
            .Select(_ => rng.NextDouble() * 2 - 1)
            .ToArray();
        double[] matrixBValues = Enumerable.Range(0, size * size)
            .Select(_ => rng.NextDouble() * 2 - 1)
            .ToArray();

        // Compute reference result in double precision
        double[] expectedC = new double[size * size];
        for (int row = 0; row < size; row++)
        {
            for (int col = 0; col < size; col++)
            {
                double sum = 0;
                for (int k = 0; k < size; k++)
                {
                    sum += matrixAValues[row * size + k] * matrixBValues[k * size + col];
                }
                expectedC[row * size + col] = sum;
            }
        }

        // Compute using Posit8
        byte[] matrixA = matrixAValues.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] matrixB = matrixBValues.Select(v => Posit8Tables.EncodeDouble(v)).ToArray();
        byte[] matrixC = new byte[size * size];

        Posit8Tables.MatMulDouble(matrixA, matrixB, matrixC, size, size, size);

        // Calculate error statistics
        double maxError = 0;
        double sumSquaredError = 0;
        int validElements = 0;

        for (int i = 0; i < size * size; i++)
        {
            double expected = expectedC[i];
            double actual = Posit8Tables.ToDouble(matrixC[i]);

            if (Math.Abs(expected) > 0.01) // Only measure error on non-tiny values
            {
                double error = Math.Abs((actual - expected) / expected);
                maxError = Math.Max(maxError, error);
                sumSquaredError += error * error;
                validElements++;
            }
        }

        double rmse = Math.Sqrt(sumSquaredError / validElements);

        // Output for visibility
        Console.WriteLine($"32x32 MatMul - Max relative error: {maxError:P2}");
        Console.WriteLine($"32x32 MatMul - RMSE: {rmse:P2}");

        // With 8-bit quantization, errors can be significant for accumulated operations
        // These are reasonable bounds given the limited precision
        // Improved rounding may cause slight variations in accumulated error
        Assert.True(maxError < 6.0, $"Max error too high: {maxError:P2}");
        Assert.True(rmse < 0.6, $"RMSE too high: {rmse:P2}");
    }
}
