using System;

namespace Posit8.Net;

/// <summary>
/// Base exception for all Posit8-related errors.
/// </summary>
public class Posit8Exception : Exception
{
    public Posit8Exception(string message) : base(message) { }
    public Posit8Exception(string message, Exception innerException) : base(message, innerException) { }
}

/// <summary>
/// Exception thrown when OpenCL initialization or operations fail.
/// </summary>
public class Posit8OpenCLException : Posit8Exception
{
    public string? ErrorCode { get; }

    public Posit8OpenCLException(string message, string? errorCode = null)
        : base(message)
    {
        ErrorCode = errorCode;
    }

    public Posit8OpenCLException(string message, string? errorCode, Exception innerException)
        : base(message, innerException)
    {
        ErrorCode = errorCode;
    }
}

/// <summary>
/// Exception thrown when posit operations overflow or underflow.
/// </summary>
public class Posit8OverflowException : Posit8Exception
{
    public Posit8OverflowException(string message) : base(message) { }
}

/// <summary>
/// Exception thrown when invalid operations are attempted (e.g., division by zero).
/// </summary>
public class Posit8InvalidOperationException : Posit8Exception
{
    public Posit8InvalidOperationException(string message) : base(message) { }
}

/// <summary>
/// Exception thrown when matrix dimensions are invalid.
/// </summary>
public class Posit8DimensionException : Posit8Exception
{
    public int ExpectedLength { get; }
    public int ActualLength { get; }

    public Posit8DimensionException(string message, int expected, int actual)
        : base(message)
    {
        ExpectedLength = expected;
        ActualLength = actual;
    }
}
