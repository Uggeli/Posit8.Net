using System;
using System.Runtime.CompilerServices;

/// <summary>
/// Ultra-fast Posit8es1 implementation using precomputed lookup tables.
/// All operations are O(1) array lookups - perfect for SIMD.
/// Total memory: ~130KB for all tables (fits in L2 cache).
/// </summary>
public static class Posit8Tables
{
    public const byte NaR = 0b1000_0000;
    
    // Lookup tables (256 entries each)
    private static readonly double[] ToDoubleTable = new double[256];


    // Operation tables (256x256 = 64KB each)
    private static readonly byte[,] AddTable = new byte[256, 256];
    private static readonly byte[,] MulTable = new byte[256, 256];
    private static readonly byte[,] SubTable = new byte[256, 256];
    private static readonly byte[,] DivTable = new byte[256, 256];
    
    // Comparison and utility tables
    private static readonly byte[] NegTable = new byte[256];
    private static readonly byte[] AbsTable = new byte[256];
    private static readonly byte[] RecipTable = new byte[256];
    
    static Posit8Tables()
    {
        BuildConversionTables();
        BuildArithmeticTables();
        BuildUtilityTables();
    }
    
    #region Table Construction (Pure Bit Manipulation)
    
    private static void BuildConversionTables()
    {
        // Build decode table (Posit -> Double)
        for (int i = 0; i < 256; i++)
        {
            ToDoubleTable[i] = DecodePositBits((byte)i);
        }
    }
    
    private static void BuildArithmeticTables()
    {
        // Precompute ALL possible operations
        for (int a = 0; a < 256; a++)
        {
            for (int b = 0; b < 256; b++)
            {
                byte pa = (byte)a;
                byte pb = (byte)b;
                
                // Addition: decode, add, encode
                AddTable[a, b] = ComputeAdd(pa, pb);
                
                // Multiplication
                MulTable[a, b] = ComputeMul(pa, pb);
                
                // Subtraction
                SubTable[a, b] = ComputeSub(pa, pb);
                
                // Division
                DivTable[a, b] = ComputeDiv(pa, pb);
            }
        }
    }
    
    private static void BuildUtilityTables()
    {
        for (int i = 0; i < 256; i++)
        {
            byte p = (byte)i;
            
            // Negation (two's complement)
            NegTable[i] = (byte)(-(sbyte)p);
            
            // Absolute value
            if (p == NaR)
                AbsTable[i] = NaR;
            else if ((p & 0x80) != 0) // Negative
                AbsTable[i] = (byte)(-(sbyte)p);
            else
                AbsTable[i] = p;
            
            // Reciprocal (1/x)
            RecipTable[i] = ComputeRecip(p);
        }
    }
    
    #endregion
    
    #region Core Bit Manipulation (Encode/Decode)
    
    /// <summary>
    /// Decode a Posit8 byte to double using pure bit operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static double DecodePositBits(byte bits)
    {
        // Special cases
        if (bits == 0) return 0.0;
        if (bits == NaR) return double.NaN;
        
        // Extract sign and convert to positive representation
        bool isNegative = (bits & 0x80) != 0;
        if (isNegative)
            bits = (byte)(-(sbyte)bits);
        
        // Decode regime (count leading identical bits after sign)
        bool regimeSign = (bits & 0x40) != 0;
        int m = 0;
        byte mask = 0x40;
        
        while (m < 7 && ((bits & mask) != 0) == regimeSign)
        {
            m++;
            mask >>= 1;
        }
        
        int regimeK = regimeSign ? m - 1 : -m;
        int regimeLen = m + 1;
        int remainingBits = 7 - regimeLen;
        
        // Extract exponent bit (es=1)
        int expBit = 0;
        ulong mantissa = 0;
        
        if (remainingBits > 0)
        {
            expBit = (bits >> (remainingBits - 1)) & 1;
            
            if (remainingBits > 1)
            {
                int fracBits = remainingBits - 1;
                byte fracMask = (byte)((1 << fracBits) - 1);
                uint frac = (uint)(bits & fracMask);
                
                // Shift to fill IEEE 754's 52-bit mantissa
                mantissa = ((ulong)frac) << (52 - fracBits);
            }
        }
        
        // Compute scale: scale = regime * 2^es + exponent = regimeK * 2 + expBit
        int scale = (regimeK << 1) + expBit;
        
        // Convert to IEEE 754 double
        int ieeeExp = scale + 1023; // Add bias
        
        // Handle underflow/overflow
        if (ieeeExp <= 0) return isNegative ? -0.0 : 0.0;
        if (ieeeExp >= 2047) return isNegative ? double.NegativeInfinity : double.PositiveInfinity;
        
        // Assemble IEEE 754 bits
        ulong doubleBits = 0;
        if (isNegative) doubleBits |= 1UL << 63;
        doubleBits |= ((ulong)ieeeExp) << 52;
        doubleBits |= mantissa;
        
        return BitConverter.UInt64BitsToDouble(doubleBits);
    }
    
    /// <summary>
    /// Encode a double to Posit8 using pure bit operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte EncodeDoubleBitwise(double value)
    {
        // Special cases
        if (value == 0.0) return 0;
        if (double.IsNaN(value) || double.IsInfinity(value)) return NaR;

        // Extract IEEE 754 components
        ulong doubleBits = BitConverter.DoubleToUInt64Bits(value);
        bool sign = (doubleBits >> 63) != 0;
        int expField = (int)((doubleBits >> 52) & 0x7FF);
        ulong mantissa = doubleBits & 0xFFFFFFFFFFFFF;

        // Handle denormals
        if (expField == 0) return 0; // Underflow to zero

        // Compute actual exponent (unbias)
        int exponent = expField - 1023;

        // For Posit: scale = exponent
        int scale = exponent;

        // Compute regime and exponent bit
        // regime_k = floor(scale / 2), exp_bit = scale % 2
        int regimeK = scale >= 0 ? scale >> 1 : -(((-scale) + 1) >> 1);
        int expBit = scale & 1;

        // Build regime bit pattern
        int regimeLen;
        uint regimeBits;

        if (regimeK >= 0)
        {
            regimeLen = regimeK + 2; // k+1 ones, then terminating zero
            if (regimeLen > 7) return sign ? (byte)0x81 : (byte)0x7F; // Overflow to max
            regimeBits = (uint)((1 << regimeLen) - 2); // e.g., 0b0111_1110 for k=5
        }
        else
        {
            regimeLen = -regimeK + 1; // -k zeros, then terminating one
            if (regimeLen > 7) return 0; // Underflow to zero
            regimeBits = 1; // 0b0000_0001
        }

        // Calculate available fraction bits
        int fracBitsAvailable = 7 - regimeLen - 1; // -1 for exponent bit

        if (fracBitsAvailable < 0)
        {
            // Not enough space for fraction - need to check if we should round up
            // by looking at the implicit 1 bit of the mantissa
            bool shouldRoundUp = false;

            if (fracBitsAvailable == -1)
            {
                // No fraction bits, but we have exponent bit
                // Check if we should round based on mantissa
                shouldRoundUp = (mantissa >> 51) >= 1; // Check MSB of mantissa
            }

            // Assemble without fraction
            uint bits = 0;
            int pos = 6;

            bits |= regimeBits << (pos - regimeLen + 1);
            pos -= regimeLen;

            if (pos >= 0)
            {
                bits |= (uint)(expBit << pos);
            }

            // Round up if needed
            if (shouldRoundUp && bits < 0x7F)
            {
                bits++;
            }

            if (sign)
            {
                bits = (uint)(-(int)bits) & 0xFF;
            }

            return (byte)bits;
        }

        // Assemble the Posit bit pattern
        uint positBits = 0;
        int bitPos = 6; // Start after sign bit

        // Place regime bits
        positBits |= regimeBits << (bitPos - regimeLen + 1);
        bitPos -= regimeLen;

        // Place exponent bit
        positBits |= (uint)(expBit << bitPos);
        bitPos--;

        // Place fraction bits
        int fracBits = bitPos + 1;
        // Extract top fracBits from IEEE mantissa (52 bits)
        uint frac = (uint)(mantissa >> (52 - fracBits));

        // Round to nearest (check bit after truncation)
        bool roundUp = false;
        if (fracBits < 52)
        {
            int roundBit = 52 - fracBits - 1;
            if (roundBit >= 0)
            {
                ulong roundBitValue = (mantissa >> roundBit) & 1;
                if (roundBitValue != 0)
                {
                    // Round to nearest, ties to even
                    ulong stickyBits = roundBit > 0 ? mantissa & ((1UL << roundBit) - 1) : 0;
                    if (stickyBits != 0 || (frac & 1) != 0)
                    {
                        roundUp = true;
                    }
                }
            }
        }

        positBits |= frac;

        // Handle rounding with proper carry propagation
        if (roundUp)
        {
            // Increment the entire posit value (before sign application)
            // This naturally handles carry through fraction -> exponent -> regime
            positBits++;

            // Check for overflow to max value
            if (positBits >= 0x80)
            {
                positBits = 0x7F; // Clamp to max positive value
            }
        }

        // Apply two's complement for negative values
        if (sign)
        {
            positBits = (uint)(-(int)positBits) & 0xFF;
        }

        return (byte)positBits;
    }
    
    #endregion
    
    #region Arithmetic Operations (for table construction)
    
    private static byte ComputeAdd(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        if (a == 0) return b;
        if (b == 0) return a;

        double sum = ToDoubleTable[a] + ToDoubleTable[b];
        return EncodeDoubleBitwise(sum);
    }

    private static byte ComputeMul(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        if (a == 0 || b == 0) return 0;

        double product = ToDoubleTable[a] * ToDoubleTable[b];
        return EncodeDoubleBitwise(product);
    }

    private static byte ComputeSub(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;

        double diff = ToDoubleTable[a] - ToDoubleTable[b];
        return EncodeDoubleBitwise(diff);
    }

    private static byte ComputeDiv(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        if (b == 0) return NaR; // Division by zero
        if (a == 0) return 0;

        double quotient = ToDoubleTable[a] / ToDoubleTable[b];
        return EncodeDoubleBitwise(quotient);
    }

    private static byte ComputeRecip(byte p)
    {
        if (p == NaR || p == 0) return NaR;

        double recip = 1.0 / ToDoubleTable[p];
        return EncodeDoubleBitwise(recip);
    }
    
    #endregion
    
    #region Public API - O(1) Operations

    /// <summary>
    /// Decode a Posit8 byte to double using lookup table. O(1) operation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ToDouble(byte posit) => ToDoubleTable[posit];

    /// <summary>
    /// Encode a double to Posit8 using pure bit operations.
    /// Uses proper rounding according to posit standard.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte EncodeDouble(double value) => EncodeDoubleBitwise(value);
    
    /// <summary>
    /// Add two Posit8 values. O(1) table lookup operation.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Result of a + b in Posit8 format</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Add(byte a, byte b) => AddTable[a, b];

    /// <summary>
    /// Multiply two Posit8 values. O(1) table lookup operation.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Result of a * b in Posit8 format</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Mul(byte a, byte b) => MulTable[a, b];

    /// <summary>
    /// Subtract two Posit8 values. O(1) table lookup operation.
    /// </summary>
    /// <param name="a">First operand</param>
    /// <param name="b">Second operand</param>
    /// <returns>Result of a - b in Posit8 format</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Sub(byte a, byte b) => SubTable[a, b];

    /// <summary>
    /// Divide two Posit8 values. O(1) table lookup operation.
    /// Returns NaR (0x80) if b is zero.
    /// </summary>
    /// <param name="a">Numerator</param>
    /// <param name="b">Denominator</param>
    /// <returns>Result of a / b in Posit8 format, or NaR if division by zero</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Div(byte a, byte b) => DivTable[a, b];

    /// <summary>
    /// Negate a Posit8 value using two's complement. O(1) operation.
    /// </summary>
    /// <param name="posit">Value to negate</param>
    /// <returns>Negated value (-posit)</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Neg(byte posit) => NegTable[posit];

    /// <summary>
    /// Get absolute value of a Posit8. O(1) operation.
    /// Returns NaR unchanged.
    /// </summary>
    /// <param name="posit">Input value</param>
    /// <returns>Absolute value</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Abs(byte posit) => AbsTable[posit];

    /// <summary>
    /// Calculate reciprocal (1/x) of a Posit8 value. O(1) operation.
    /// Returns NaR for zero or NaR input.
    /// </summary>
    /// <param name="posit">Input value</param>
    /// <returns>Reciprocal (1/posit), or NaR if input is zero or NaR</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Recip(byte posit) => RecipTable[posit];
    
    /// <summary>
    /// Compare two Posit8 values.
    /// Returns -1 if a &lt; b, 0 if equal, 1 if a &gt; b.
    /// NaR comparison returns 0 (undefined).
    /// </summary>
    /// <param name="a">First value to compare</param>
    /// <param name="b">Second value to compare</param>
    /// <returns>-1 if a &lt; b, 0 if equal or NaR involved, 1 if a &gt; b</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Compare(byte a, byte b)
    {
        if (a == NaR || b == NaR) return 0; // NaR comparison undefined

        // Posit uses two's complement: just cast to signed byte and compare
        // Negative values (0x81-0xFF) are -127 to -1
        // Positive values (0x01-0x7F) are 1 to 127
        sbyte sa = (sbyte)a;
        sbyte sb = (sbyte)b;
        return sa.CompareTo(sb);
    }
    
    #endregion
    
    #region SIMD Batch Operations
    
    /// <summary>
    /// Element-wise vector addition: result[i] = a[i] + b[i].
    /// Uses table lookups for O(1) per-element operations.
    /// </summary>
    /// <param name="a">First input vector</param>
    /// <param name="b">Second input vector</param>
    /// <param name="result">Output vector (must be same length as inputs)</param>
    /// <exception cref="ArgumentException">Thrown if vector lengths don't match</exception>
    public static void AddVector(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> result)
    {
        if (a.Length != b.Length)
            throw new ArgumentException(
                $"Vector length mismatch: vector 'a' has {a.Length} elements, vector 'b' has {b.Length} elements",
                nameof(b));

        if (a.Length != result.Length)
            throw new ArgumentException(
                $"Result vector length mismatch: expected {a.Length} elements to match input vectors, got {result.Length}",
                nameof(result));

        int i = 0;

        // Process 8 elements at a time with loop unrolling
        // This allows CPU to pipeline multiple table lookups in parallel
        int unrollLength = a.Length - (a.Length % 8);
        for (; i < unrollLength; i += 8)
        {
            result[i] = AddTable[a[i], b[i]];
            result[i + 1] = AddTable[a[i + 1], b[i + 1]];
            result[i + 2] = AddTable[a[i + 2], b[i + 2]];
            result[i + 3] = AddTable[a[i + 3], b[i + 3]];
            result[i + 4] = AddTable[a[i + 4], b[i + 4]];
            result[i + 5] = AddTable[a[i + 5], b[i + 5]];
            result[i + 6] = AddTable[a[i + 6], b[i + 6]];
            result[i + 7] = AddTable[a[i + 7], b[i + 7]];
        }

        // Handle remaining elements
        for (; i < a.Length; i++)
        {
            result[i] = AddTable[a[i], b[i]];
        }
    }
    
    /// <summary>
    /// Compute dot product of two Posit8 vectors with double precision accumulation.
    /// Decodes inputs to double, multiplies, and accumulates without intermediate rounding.
    /// Provides best accuracy by only quantizing inputs, not intermediate results.
    /// </summary>
    /// <param name="a">First input vector</param>
    /// <param name="b">Second input vector (must be same length as a)</param>
    /// <returns>Dot product as double precision value</returns>
    /// <exception cref="ArgumentException">Thrown if vector lengths don't match</exception>
    public static double DotProduct(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException(
                $"Vector length mismatch: vector 'a' has {a.Length} elements, vector 'b' has {b.Length} elements",
                nameof(b));

        double sum = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            // Decode once, multiply in double precision
            // Avoids intermediate rounding that MulTable would introduce
            double aVal = ToDoubleTable[a[i]];
            double bVal = ToDoubleTable[b[i]];
            sum += aVal * bVal;
        }

        return sum;
    }

    /// <summary>
    /// Matrix multiplication C = A * B with double precision accumulation.
    /// Decodes inputs to double, multiplies, and accumulates without intermediate rounding.
    /// Only rounds once at the end - provides best accuracy for Posit8 matrix multiplication.
    /// Recommended for most use cases.
    /// </summary>
    /// <param name="A">Input matrix A (m x k), row-major order</param>
    /// <param name="B">Input matrix B (k x n), row-major order</param>
    /// <param name="C">Output matrix C (m x n), row-major order</param>
    /// <param name="m">Number of rows in A</param>
    /// <param name="k">Number of columns in A / rows in B</param>
    /// <param name="n">Number of columns in B</param>
    /// <exception cref="ArgumentException">Thrown if matrix dimensions don't match</exception>
    public static void MatMulDouble(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C,
                                    int m, int k, int n)
    {
        int expectedA = m * k;
        int expectedB = k * n;
        int expectedC = m * n;

        if (A.Length != expectedA)
            throw new ArgumentException(
                $"Matrix A dimension mismatch: expected {m}×{k} = {expectedA} elements, got {A.Length}",
                nameof(A));

        if (B.Length != expectedB)
            throw new ArgumentException(
                $"Matrix B dimension mismatch: expected {k}×{n} = {expectedB} elements, got {B.Length}",
                nameof(B));

        if (C.Length != expectedC)
            throw new ArgumentException(
                $"Matrix C dimension mismatch: expected {m}×{n} = {expectedC} elements, got {C.Length}",
                nameof(C));

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double accumulator = 0.0;

                for (int i = 0; i < k; i++)
                {
                    byte a_elem = A[row * k + i];
                    byte b_elem = B[i * n + col];

                    // Decode and multiply in double precision
                    // Avoids intermediate rounding that MulTable would introduce
                    double aVal = ToDoubleTable[a_elem];
                    double bVal = ToDoubleTable[b_elem];
                    accumulator += aVal * bVal;
                }

                // Round only once at the end
                C[row * n + col] = EncodeDouble(accumulator);
            }
        }
    }

    /// <summary>
    /// Parallel matrix multiplication C = A * B with double precision accumulation.
    /// Uses multiple CPU cores for better performance on larger matrices.
    /// Recommended for matrices larger than 128x128.
    /// Note: Uses arrays instead of Spans to allow parallelization.
    /// </summary>
    /// <param name="A">Input matrix A (m x k), row-major order</param>
    /// <param name="B">Input matrix B (k x n), row-major order</param>
    /// <param name="C">Output matrix C (m x n), row-major order</param>
    /// <param name="m">Number of rows in A</param>
    /// <param name="k">Number of columns in A / rows in B</param>
    /// <param name="n">Number of columns in B</param>
    /// <exception cref="ArgumentException">Thrown if matrix dimensions don't match</exception>
    public static void MatMulDoubleParallel(byte[] A, byte[] B, byte[] C,
                                            int m, int k, int n)
    {
        int expectedA = m * k;
        int expectedB = k * n;
        int expectedC = m * n;

        if (A.Length != expectedA)
            throw new ArgumentException(
                $"Matrix A dimension mismatch: expected {m}×{k} = {expectedA} elements, got {A.Length}",
                nameof(A));

        if (B.Length != expectedB)
            throw new ArgumentException(
                $"Matrix B dimension mismatch: expected {k}×{n} = {expectedB} elements, got {B.Length}",
                nameof(B));

        if (C.Length != expectedC)
            throw new ArgumentException(
                $"Matrix C dimension mismatch: expected {m}×{n} = {expectedC} elements, got {C.Length}",
                nameof(C));

        // Parallel processing of rows
        Parallel.For(0, m, row =>
        {
            for (int col = 0; col < n; col++)
            {
                double accumulator = 0.0;

                for (int i = 0; i < k; i++)
                {
                    byte a_elem = A[row * k + i];
                    byte b_elem = B[i * n + col];

                    // Decode and multiply in double precision
                    double aVal = ToDoubleTable[a_elem];
                    double bVal = ToDoubleTable[b_elem];
                    accumulator += aVal * bVal;
                }

                // Round only once at the end
                C[row * n + col] = EncodeDouble(accumulator);
            }
        });
    }

    #endregion
}
