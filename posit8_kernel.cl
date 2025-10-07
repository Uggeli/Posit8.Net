// Posit8 Matrix Multiplication OpenCL Kernel
// Lookup tables passed as kernel arguments

// Encode double to posit8 (simplified version for GPU)
uchar encode_double(double value) {
    if (value == 0.0) return 0;
    if (isnan(value) || isinf(value)) return 0x80; // NaR

    bool sign = value < 0.0;
    if (sign) value = -value;

    // Get IEEE 754 components
    ulong bits = as_ulong(value);
    int expField = (int)((bits >> 52) & 0x7FF);

    if (expField == 0) return 0; // Denormal -> underflow

    int exponent = expField - 1023;
    int scale = exponent;

    int regimeK = scale >> 1;
    int expBit = scale & 1;

    // Build regime
    int regimeLen;
    uint regimeBits;

    if (regimeK >= 0) {
        regimeLen = regimeK + 2;
        if (regimeLen > 7) return sign ? 0x81 : 0x7F;
        regimeBits = (1 << regimeLen) - 2;
    } else {
        regimeLen = -regimeK + 1;
        if (regimeLen > 7) return 0;
        regimeBits = 1;
    }

    int fracBitsAvailable = 7 - regimeLen - 1;
    if (fracBitsAvailable < -1) return 0;

    // Assemble posit
    uint positBits = 0;
    int bitPos = 6;

    positBits |= regimeBits << (bitPos - regimeLen + 1);
    bitPos -= regimeLen;

    if (bitPos >= 0) {
        positBits |= (uint)(expBit << bitPos);
        bitPos--;
    }

    if (bitPos >= 0) {
        int fracBits = bitPos + 1;
        ulong mantissa = bits & 0xFFFFFFFFFFFFF;
        uint frac = (uint)(mantissa >> (52 - fracBits));
        positBits |= frac;
    }

    if (sign) {
        positBits = (-(int)positBits) & 0xFF;
    }

    return (uchar)positBits;
}

// Matrix multiplication kernel with double accumulation
__kernel void matmul_posit8_double(
    __global const uchar* A,
    __global const uchar* B,
    __global uchar* C,
    int M, int N, int K,
    __constant uchar* MulTable,
    __constant double* ToDoubleTable)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= M || col >= N) return;

    double acc = 0.0;

    for (int k = 0; k < K; k++) {
        uchar a = A[row * K + k];
        uchar b = B[k * N + col];

        // Table lookup: MulTable is indexed as [a][b] = a * 256 + b
        uchar prod = MulTable[a * 256 + b];
        acc += ToDoubleTable[prod];
    }

    C[row * N + col] = encode_double(acc);
}

// Matrix multiplication kernel with byte accumulation (faster but less accurate)
__kernel void matmul_posit8_byte(
    __global const uchar* A,
    __global const uchar* B,
    __global uchar* C,
    int M, int N, int K,
    __constant uchar* MulTable,
    __constant uchar* AddTable)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= M || col >= N) return;

    uchar acc = 0;

    for (int k = 0; k < K; k++) {
        uchar a = A[row * K + k];
        uchar b = B[k * N + col];

        uchar prod = MulTable[a * 256 + b];
        acc = AddTable[acc * 256 + prod];
    }

    C[row * N + col] = acc;
}
