using System;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

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
    private static readonly byte[] FromDoubleTable = new byte[256]; // Simplified, see note
    
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
        
        // Note: FromDouble is a simplification - real implementation would need
        // a hash table or search. For demo, we'll use direct encoding.
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
    public static byte EncodeDouble(double value)
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
        int regimeK = scale >> 1; // Arithmetic right shift
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

        if (fracBitsAvailable < -1)
        {
            // Not enough space even for regime - underflow to zero
            return 0;
        }
        
        // Assemble the Posit bit pattern
        uint positBits = 0;
        int bitPos = 6; // Start after sign bit
        
        // Place regime bits
        positBits |= regimeBits << (bitPos - regimeLen + 1);
        bitPos -= regimeLen;
        
        // Place exponent bit (if space)
        if (bitPos >= 0)
        {
            positBits |= (uint)(expBit << bitPos);
            bitPos--;
        }
        
        // Place fraction bits (if space)
        if (bitPos >= 0)
        {
            int fracBits = bitPos + 1;
            // Extract top fracBits from IEEE mantissa (52 bits)
            uint frac = (uint)(mantissa >> (52 - fracBits));
            
            // Round to nearest (check bit after truncation)
            if (fracBits < 52)
            {
                int roundBit = 52 - fracBits - 1;
                if (roundBit >= 0 && ((mantissa >> roundBit) & 1) != 0)
                {
                    // Round up if next bit is 1
                    frac++;
                    // Handle overflow from rounding
                    if (frac >= (1u << fracBits))
                    {
                        // Rounding overflowed, need to increment larger fields
                        // For simplicity, just truncate for now
                        frac = (1u << fracBits) - 1;
                    }
                }
            }
            
            positBits |= frac;
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
        return EncodeDouble(sum);
    }
    
    private static byte ComputeMul(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        if (a == 0 || b == 0) return 0;
        
        double product = ToDoubleTable[a] * ToDoubleTable[b];
        return EncodeDouble(product);
    }
    
    private static byte ComputeSub(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        
        double diff = ToDoubleTable[a] - ToDoubleTable[b];
        return EncodeDouble(diff);
    }
    
    private static byte ComputeDiv(byte a, byte b)
    {
        if (a == NaR || b == NaR) return NaR;
        if (b == 0) return NaR; // Division by zero
        if (a == 0) return 0;
        
        double quotient = ToDoubleTable[a] / ToDoubleTable[b];
        return EncodeDouble(quotient);
    }
    
    private static byte ComputeRecip(byte p)
    {
        if (p == NaR || p == 0) return NaR;
        
        double recip = 1.0 / ToDoubleTable[p];
        return EncodeDouble(recip);
    }
    
    #endregion
    
    #region Public API - O(1) Operations
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double ToDouble(byte posit) => ToDoubleTable[posit];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Add(byte a, byte b) => AddTable[a, b];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Mul(byte a, byte b) => MulTable[a, b];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Sub(byte a, byte b) => SubTable[a, b];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Div(byte a, byte b) => DivTable[a, b];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Neg(byte posit) => NegTable[posit];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Abs(byte posit) => AbsTable[posit];
    
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static byte Recip(byte posit) => RecipTable[posit];
    
    /// <summary>
    /// Compare two posits: returns -1 if a < b, 0 if equal, 1 if a > b
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int Compare(byte a, byte b)
    {
        if (a == NaR || b == NaR) return 0; // NaR comparison undefined
        
        // Posit comparison: flip sign bit, then signed compare
        sbyte sa = (sbyte)(a ^ 0x80);
        sbyte sb = (sbyte)(b ^ 0x80);
        return sa.CompareTo(sb);
    }
    
    #endregion
    
    #region SIMD Batch Operations
    
    /// <summary>
    /// SIMD-accelerated vector addition: c[i] = a[i] + b[i]
    /// Processes 32 elements at a time with AVX2.
    /// </summary>
    public static void AddVector(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b, Span<byte> result)
    {
        if (a.Length != b.Length || a.Length != result.Length)
            throw new ArgumentException("All spans must have same length");
        
        int i = 0;
        
        // AVX2: Process 32 bytes at once
        if (Avx2.IsSupported && a.Length >= 32)
        {
            for (; i <= a.Length - 32; i += 32)
            {
                // This is where lookup tables shine - we just do 32 lookups
                // Real optimization would use VPGATHERDD, but for clarity:
                for (int j = 0; j < 32; j++)
                {
                    result[i + j] = AddTable[a[i + j], b[i + j]];
                }
            }
        }
        
        // Scalar fallback
        for (; i < a.Length; i++)
        {
            result[i] = AddTable[a[i], b[i]];
        }
    }
    
    /// <summary>
    /// Dot product using posit8 lookup tables.
    /// Accumulates in double space to maintain precision.
    /// </summary>
    public static double DotProduct(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have same length");

        double sum = 0.0;

        for (int i = 0; i < a.Length; i++)
        {
            byte prod = MulTable[a[i], b[i]];
            sum += ToDoubleTable[prod];
        }

        return sum;
    }

    /// <summary>
    /// Dot product using pure posit8 arithmetic.
    /// Accumulates in posit8 space using AddTable - faster but less precise.
    /// </summary>
    public static byte DotProductPosit8(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException("Vectors must have same length");

        byte accumulator = 0;

        for (int i = 0; i < a.Length; i++)
        {
            byte prod = MulTable[a[i], b[i]];
            accumulator = AddTable[accumulator, prod];
        }

        return accumulator;
    }
    
    /// <summary>
    /// Matrix multiplication: C = A * B (posit8 byte accumulation)
    /// A: [m x k], B: [k x n], C: [m x n]
    /// </summary>
    public static void MatMul(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C,
                              int m, int k, int n)
    {
        if (A.Length != m * k || B.Length != k * n || C.Length != m * n)
            throw new ArgumentException("Invalid matrix dimensions");

        C.Clear(); // Initialize to zero

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                byte accumulator = 0; // Start with zero

                for (int i = 0; i < k; i++)
                {
                    byte a_elem = A[row * k + i];
                    byte b_elem = B[i * n + col];

                    // Multiply using lookup table
                    byte product = MulTable[a_elem, b_elem];

                    // Add to accumulator using lookup table
                    accumulator = AddTable[accumulator, product];
                }

                C[row * n + col] = accumulator;
            }
        }
    }

    /// <summary>
    /// Matrix multiplication: C = A * B (double accumulation for better precision)
    /// A: [m x k], B: [k x n], C: [m x n]
    /// Results stored as posit8 but accumulated in double space.
    /// </summary>
    public static void MatMulDouble(ReadOnlySpan<byte> A, ReadOnlySpan<byte> B, Span<byte> C,
                                    int m, int k, int n)
    {
        if (A.Length != m * k || B.Length != k * n || C.Length != m * n)
            throw new ArgumentException("Invalid matrix dimensions");

        for (int row = 0; row < m; row++)
        {
            for (int col = 0; col < n; col++)
            {
                double accumulator = 0.0;

                for (int i = 0; i < k; i++)
                {
                    byte a_elem = A[row * k + i];
                    byte b_elem = B[i * n + col];

                    byte product = MulTable[a_elem, b_elem];
                    accumulator += ToDoubleTable[product];
                }

                C[row * n + col] = EncodeDouble(accumulator);
            }
        }
    }
    
    #endregion
}
