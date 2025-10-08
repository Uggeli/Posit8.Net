using System;
using System.IO;
using System.Runtime.InteropServices;
using OpenCL.Net;

namespace Posit8.Net;

/// <summary>
/// OpenCL GPU-accelerated matrix operations for Posit8.
/// Implements IDisposable for proper resource cleanup.
/// </summary>
public sealed class Posit8OpenCL : IDisposable
{
    private Context context;
    private Device device;
    private CommandQueue queue;
    private Kernel kernel;
    private Mem toDoubleTableBuffer;
    private bool disposed = false;

    // Device capability limits
    private ulong maxMemAllocSize;
    private ulong globalMemSize;
    private long maxWorkGroupSize;

    /// <summary>
    /// Initialize OpenCL for Posit8 matrix operations.
    /// </summary>
    /// <exception cref="Posit8OpenCLException">Thrown when OpenCL initialization fails.</exception>
    public Posit8OpenCL()
    {
        ErrorCode error;

        try
        {
            // Get platform
            Platform[] platforms = new Platform[16];
            uint platformCount = 0;
            error = Cl.GetPlatformIDs(0, null, out platformCount);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to get OpenCL platform count", error.ToString());

            error = Cl.GetPlatformIDs(platformCount, platforms, out platformCount);
            if (error != ErrorCode.Success || platformCount == 0)
                throw new Posit8OpenCLException("No OpenCL platforms found", error.ToString());

            Platform platform = platforms[0];

            // Get device (prefer GPU, fallback to CPU)
            Device[] devices = new Device[16];
            uint deviceCount = 0;
            error = Cl.GetDeviceIDs(platform, DeviceType.Gpu, 0, null, out deviceCount);
            error = Cl.GetDeviceIDs(platform, DeviceType.Gpu, deviceCount, devices, out deviceCount);

            if (deviceCount == 0)
            {
                // Try CPU
                error = Cl.GetDeviceIDs(platform, DeviceType.Cpu, 0, null, out deviceCount);
                error = Cl.GetDeviceIDs(platform, DeviceType.Cpu, deviceCount, devices, out deviceCount);
                if (error != ErrorCode.Success || deviceCount == 0)
                    throw new Posit8OpenCLException("No OpenCL devices (GPU or CPU) found", error.ToString());
            }

            device = devices[0];

            // Query device capabilities
            InfoBuffer deviceInfo;

            deviceInfo = Cl.GetDeviceInfo(device, DeviceInfo.MaxMemAllocSize, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to query device max memory allocation size", error.ToString());
            maxMemAllocSize = deviceInfo.CastTo<ulong>();

            deviceInfo = Cl.GetDeviceInfo(device, DeviceInfo.GlobalMemSize, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to query device global memory size", error.ToString());
            globalMemSize = deviceInfo.CastTo<ulong>();

            deviceInfo = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to query device max work group size", error.ToString());
            maxWorkGroupSize = deviceInfo.CastTo<long>();

            // Create context
            context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to create OpenCL context", error.ToString());

            // Create command queue
            queue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to create command queue", error.ToString());

            // Read and build kernel from embedded resource
            string kernelSource;
            var assembly = typeof(Posit8OpenCL).Assembly;
            using (var stream = assembly.GetManifestResourceStream("Posit8.Net.posit8_kernel.cl"))
            {
                if (stream == null)
                    throw new Posit8OpenCLException("Could not find embedded kernel resource 'Posit8.Net.posit8_kernel.cl'. Ensure the file is marked as EmbeddedResource.");

                using (var reader = new StreamReader(stream))
                {
                    kernelSource = reader.ReadToEnd();
                }
            }

            OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to create OpenCL program", error.ToString());

            error = Cl.BuildProgram(program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            if (error != ErrorCode.Success)
            {
                InfoBuffer buildLog = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out error);
                throw new Posit8OpenCLException($"Failed to build OpenCL kernel: {buildLog}", error.ToString());
            }

            // Create kernel
            kernel = Cl.CreateKernel(program, "matmul_posit8", out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to create kernel 'matmul_posit8'", error.ToString());

            // Prepare lookup table for decoding
            double[] toDoubleTable = new double[256];
            for (int i = 0; i < 256; i++)
            {
                toDoubleTable[i] = Posit8Tables.ToDouble((byte)i);
            }

            // Create buffer
            toDoubleTableBuffer = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(toDoubleTable.Length * sizeof(double)), toDoubleTable, out error);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to create toDoubleTable buffer", error.ToString());
        }
        catch
        {
            // Cleanup any partially initialized resources
            Dispose();
            throw;
        }
    }

    /// <summary>
    /// Perform matrix multiplication on GPU: C = A * B
    /// Uses double precision accumulation for best accuracy.
    /// </summary>
    /// <param name="A">Input matrix A (m x k)</param>
    /// <param name="B">Input matrix B (k x n)</param>
    /// <param name="C">Output matrix C (m x n)</param>
    /// <param name="M">Number of rows in A</param>
    /// <param name="N">Number of columns in B</param>
    /// <param name="K">Number of columns in A / rows in B</param>
    /// <exception cref="ObjectDisposedException">Thrown if object has been disposed.</exception>
    /// <exception cref="Posit8DimensionException">Thrown if matrix dimensions don't match.</exception>
    /// <exception cref="Posit8OpenCLException">Thrown if GPU operation fails.</exception>
    public void MatMul(byte[] A, byte[] B, byte[] C, int M, int N, int K)
    {
        if (disposed)
            throw new ObjectDisposedException(nameof(Posit8OpenCL));

        if (A.Length != M * K)
            throw new Posit8DimensionException($"Matrix A dimension mismatch", M * K, A.Length);
        if (B.Length != K * N)
            throw new Posit8DimensionException($"Matrix B dimension mismatch", K * N, B.Length);
        if (C.Length != M * N)
            throw new Posit8DimensionException($"Matrix C dimension mismatch", M * N, C.Length);

        // Validate against device capabilities
        ulong bufferSizeA = (ulong)A.Length;
        ulong bufferSizeB = (ulong)B.Length;
        ulong bufferSizeC = (ulong)C.Length;
        ulong totalBufferSize = bufferSizeA + bufferSizeB + bufferSizeC;

        if (bufferSizeA > maxMemAllocSize)
            throw new Posit8OpenCLException($"Matrix A size ({bufferSizeA} bytes) exceeds device max allocation size ({maxMemAllocSize} bytes)");
        if (bufferSizeB > maxMemAllocSize)
            throw new Posit8OpenCLException($"Matrix B size ({bufferSizeB} bytes) exceeds device max allocation size ({maxMemAllocSize} bytes)");
        if (bufferSizeC > maxMemAllocSize)
            throw new Posit8OpenCLException($"Matrix C size ({bufferSizeC} bytes) exceeds device max allocation size ({maxMemAllocSize} bytes)");

        // Check total memory including lookup table (256 * 8 = 2048 bytes)
        ulong totalMemoryNeeded = totalBufferSize + 2048;
        if (totalMemoryNeeded > globalMemSize)
            throw new Posit8OpenCLException($"Total memory required ({totalMemoryNeeded} bytes) exceeds device global memory ({globalMemSize} bytes)");

        // Validate work group size
        long totalWorkItems = (long)M * N;
        if (totalWorkItems > maxWorkGroupSize * 65536) // Reasonable limit for 2D work
            throw new Posit8OpenCLException($"Matrix dimensions too large for device: {M}x{N} work items exceeds device capabilities");

        ErrorCode error;

        // Create device buffers
        Mem bufA = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            (IntPtr)A.Length, A, out error);
        if (error != ErrorCode.Success)
            throw new Posit8OpenCLException($"Failed to create buffer for matrix A", error.ToString());

        Mem bufB = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            (IntPtr)B.Length, B, out error);
        if (error != ErrorCode.Success)
        {
            Cl.ReleaseMemObject(bufA);
            throw new Posit8OpenCLException($"Failed to create buffer for matrix B", error.ToString());
        }

        Mem bufC = (Mem)Cl.CreateBuffer(context, MemFlags.WriteOnly,
            (IntPtr)C.Length, IntPtr.Zero, out error);
        if (error != ErrorCode.Success)
        {
            Cl.ReleaseMemObject(bufA);
            Cl.ReleaseMemObject(bufB);
            throw new Posit8OpenCLException($"Failed to create buffer for matrix C", error.ToString());
        }

        try
        {
            // Set kernel arguments
            Cl.SetKernelArg(kernel, 0, bufA);
            Cl.SetKernelArg(kernel, 1, bufB);
            Cl.SetKernelArg(kernel, 2, bufC);
            Cl.SetKernelArg(kernel, 3, M);
            Cl.SetKernelArg(kernel, 4, N);
            Cl.SetKernelArg(kernel, 5, K);
            Cl.SetKernelArg(kernel, 6, toDoubleTableBuffer);

            // Execute kernel
            IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)M, (IntPtr)N };
            Event clevent;
            error = Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, null, 0, null, out clevent);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to enqueue kernel", error.ToString());

            // Wait for completion
            Cl.Finish(queue);

            // Read results
            error = Cl.EnqueueReadBuffer(queue, bufC, Bool.True, IntPtr.Zero, (IntPtr)C.Length, C, 0, null, out clevent);
            if (error != ErrorCode.Success)
                throw new Posit8OpenCLException($"Failed to read result buffer", error.ToString());
        }
        finally
        {
            // Cleanup temporary buffers
            Cl.ReleaseMemObject(bufA);
            Cl.ReleaseMemObject(bufB);
            Cl.ReleaseMemObject(bufC);
        }
    }

    /// <summary>
    /// Dispose of OpenCL resources.
    /// </summary>
    public void Dispose()
    {
        if (disposed) return;

        // Release OpenCL resources (these methods are safe to call with default/invalid handles)
        Cl.ReleaseMemObject(toDoubleTableBuffer);
        Cl.ReleaseKernel(kernel);
        Cl.ReleaseCommandQueue(queue);
        Cl.ReleaseContext(context);

        disposed = true;
        GC.SuppressFinalize(this);
    }

    /// <summary>
    /// Finalizer to ensure resources are released.
    /// </summary>
    ~Posit8OpenCL()
    {
        Dispose();
    }
}
