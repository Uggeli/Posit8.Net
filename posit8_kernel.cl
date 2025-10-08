// Posit8 Matrix Multiplication OpenCL Kernel
// Lookup tables passed as kernel arguments

// Encode double to posit8 with proper rounding
uchar encode_double(double value) {
    if (value == 0.0) return 0;
    if (isnan(value) || isinf(value)) return 0x80; // NaR

    bool sign = value < 0.0;
    if (sign) value = -value;

    // Get IEEE 754 components
    ulong bits = as_ulong(value);
    int expField = (int)((bits >> 52) & 0x7FF);
    ulong mantissa = bits & 0xFFFFFFFFFFFFFUL;

    if (expField == 0) return 0; // Denormal -> underflow

    int exponent = expField - 1023;
    int scale = exponent;

    // Compute regime K with proper floor division for negatives
    int regimeK = scale >= 0 ? scale >> 1 : -(((-scale) + 1) >> 1);
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

    if (fracBitsAvailable < 0) {
        // Not enough space for fraction - check if we should round up
        bool roundUp = false;

        if (fracBitsAvailable == -1) {
            // No fraction bits, but we have exponent bit
            // Check if we should round based on mantissa MSB
            roundUp = (mantissa >> 51) >= 1;
        }

        // Assemble without fraction
        uint positBits = 0;
        int bitPos = 6;

        positBits |= regimeBits << (bitPos - regimeLen + 1);
        bitPos -= regimeLen;

        if (bitPos >= 0) {
            positBits |= (uint)(expBit << bitPos);
        }

        // Round up if needed
        if (roundUp && positBits < 0x7F) {
            positBits++;
        }

        if (sign) {
            positBits = (-(int)positBits) & 0xFF;
        }

        return (uchar)positBits;
    }

    // Assemble posit
    uint positBits = 0;
    int bitPos = 6;

    positBits |= regimeBits << (bitPos - regimeLen + 1);
    bitPos -= regimeLen;

    positBits |= (uint)(expBit << bitPos);
    bitPos--;

    int fracBits = bitPos + 1;
    uint frac = (uint)(mantissa >> (52 - fracBits));

    // Round to nearest, ties to even
    bool roundUp = false;
    if (fracBits < 52) {
        int roundBit = 52 - fracBits - 1;
        if (roundBit >= 0) {
            ulong roundBitValue = (mantissa >> roundBit) & 1;
            if (roundBitValue != 0) {
                // Check sticky bits and LSB for tie-breaking
                ulong stickyBits = roundBit > 0 ? mantissa & ((1UL << roundBit) - 1) : 0;
                if (stickyBits != 0 || (frac & 1) != 0) {
                    roundUp = true;
                }
            }
        }
    }

    positBits |= frac;

    // Handle rounding with proper carry propagation
    if (roundUp) {
        positBits++;
        if (positBits >= 0x80) {
            positBits = 0x7F; // Clamp to max positive
        }
    }

    if (sign) {
        positBits = (-(int)positBits) & 0xFF;
    }

    return (uchar)positBits;
}

// Matrix multiplication kernel with double accumulation
// Decodes inputs, multiplies in double precision, accumulates without intermediate rounding
__kernel void matmul_posit8(
    __global const uchar* A,
    __global const uchar* B,
    __global uchar* C,
    int M, int N, int K,
    __constant double* ToDoubleTable)
{
    int row = get_global_id(0);
    int col = get_global_id(1);

    if (row >= M || col >= N) return;

    double acc = 0.0;

    for (int k = 0; k < K; k++) {
        uchar a = A[row * K + k];
        uchar b = B[k * N + col];

        // Decode and multiply in double precision
        // Avoids intermediate rounding that MulTable would introduce
        double aVal = ToDoubleTable[a];
        double bVal = ToDoubleTable[b];
        acc += aVal * bVal;
    }

    // Round only once at the end
    C[row * N + col] = encode_double(acc);
}
