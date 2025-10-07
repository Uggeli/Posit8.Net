using System;
using System.IO;
using System.Runtime.InteropServices;
using OpenCL.Net;

public static class Posit8OpenCL
{
    private static Context context;
    private static CommandQueue queue;
    private static Kernel kernelDouble;
    private static Kernel kernelByte;
    private static Mem mulTableBuffer;
    private static Mem addTableBuffer;
    private static Mem toDoubleTableBuffer;
    private static bool initialized = false;

    public static bool Initialize()
    {
        try
        {
            // Get platform
            ErrorCode error;
            Platform[] platforms = new Platform[16];
            uint platformCount = 0;
            error = Cl.GetPlatformIDs(0, null, out platformCount);
            error = Cl.GetPlatformIDs(platformCount, platforms, out platformCount);
            if (platformCount == 0)
            {
                Console.WriteLine("No OpenCL platforms found.");
                return false;
            }

            Platform platform = platforms[0];

            // Get device
            Device[] devices = new Device[16];
            uint deviceCount = 0;
            error = Cl.GetDeviceIDs(platform, DeviceType.Gpu, 0, null, out deviceCount);
            error = Cl.GetDeviceIDs(platform, DeviceType.Gpu, deviceCount, devices, out deviceCount);

            if (deviceCount == 0)
            {
                error = Cl.GetDeviceIDs(platform, DeviceType.Cpu, 0, null, out deviceCount);
                error = Cl.GetDeviceIDs(platform, DeviceType.Cpu, deviceCount, devices, out deviceCount);
                if (deviceCount == 0)
                {
                    Console.WriteLine("No OpenCL devices found.");
                    return false;
                }
            }

            Device device = devices[0];

            // Get device info
            InfoBuffer deviceName = Cl.GetDeviceInfo(device, DeviceInfo.Name, out error);
            InfoBuffer deviceType = Cl.GetDeviceInfo(device, DeviceInfo.Type, out error);
            InfoBuffer maxComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out error);
            InfoBuffer globalMemSize = Cl.GetDeviceInfo(device, DeviceInfo.GlobalMemSize, out error);

            Console.WriteLine($"Using OpenCL device: {deviceName}");
            Console.WriteLine($"  Max compute units: {maxComputeUnits.CastTo<uint>()}");
            Console.WriteLine($"  Global memory: {globalMemSize.CastTo<ulong>() / 1024 / 1024} MB");

            // Create context
            context = Cl.CreateContext(null, 1, new[] { device }, null, IntPtr.Zero, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create context: {error}");
                return false;
            }

            // Create command queue
            queue = Cl.CreateCommandQueue(context, device, CommandQueueProperties.None, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create command queue: {error}");
                return false;
            }

            // Read and build kernel from embedded resource
            string kernelSource;
            var assembly = typeof(Posit8OpenCL).Assembly;
            using (var stream = assembly.GetManifestResourceStream("Posit8.Net.posit8_kernel.cl"))
            {
                if (stream == null)
                    throw new FileNotFoundException("Could not find embedded kernel resource");

                using (var reader = new StreamReader(stream))
                {
                    kernelSource = reader.ReadToEnd();
                }
            }
            OpenCL.Net.Program program = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create program: {error}");
                return false;
            }

            error = Cl.BuildProgram(program, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            if (error != ErrorCode.Success)
            {
                InfoBuffer buildLog = Cl.GetProgramBuildInfo(program, device, ProgramBuildInfo.Log, out error);
                Console.WriteLine($"Build failed: {buildLog}");
                return false;
            }

            // Create kernels
            kernelDouble = Cl.CreateKernel(program, "matmul_posit8_double", out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create kernel double: {error}");
                return false;
            }

            kernelByte = Cl.CreateKernel(program, "matmul_posit8_byte", out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create kernel byte: {error}");
                return false;
            }

            // Prepare lookup tables
            byte[] mulTableFlat = new byte[65536];
            byte[] addTableFlat = new byte[65536];
            double[] toDoubleTable = new double[256];

            for (int i = 0; i < 256; i++)
            {
                toDoubleTable[i] = Posit8Tables.ToDouble((byte)i);
                for (int j = 0; j < 256; j++)
                {
                    mulTableFlat[i * 256 + j] = Posit8Tables.Mul((byte)i, (byte)j);
                    addTableFlat[i * 256 + j] = Posit8Tables.Add((byte)i, (byte)j);
                }
            }

            // Create buffers
            mulTableBuffer = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(mulTableFlat.Length), mulTableFlat, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create mulTable buffer: {error}");
                return false;
            }

            addTableBuffer = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(addTableFlat.Length), addTableFlat, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create addTable buffer: {error}");
                return false;
            }

            toDoubleTableBuffer = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                (IntPtr)(toDoubleTable.Length * sizeof(double)), toDoubleTable, out error);
            if (error != ErrorCode.Success)
            {
                Console.WriteLine($"Failed to create toDoubleTable buffer: {error}");
                return false;
            }

            Console.WriteLine("OpenCL initialized successfully!");
            initialized = true;
            return true;
        }
        catch (Exception ex)
        {
            Console.WriteLine($"OpenCL initialization failed: {ex.Message}");
            return false;
        }
    }

    public static void MatMulGPU(byte[] A, byte[] B, byte[] C, int M, int N, int K, bool useDoubleAccum = true)
    {
        if (!initialized)
            throw new InvalidOperationException("OpenCL not initialized");

        ErrorCode error;

        // Create device buffers
        Mem bufA = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            (IntPtr)A.Length, A, out error);
        Mem bufB = (Mem)Cl.CreateBuffer(context, MemFlags.ReadOnly | MemFlags.CopyHostPtr,
            (IntPtr)B.Length, B, out error);
        Mem bufC = (Mem)Cl.CreateBuffer(context, MemFlags.WriteOnly,
            (IntPtr)C.Length, IntPtr.Zero, out error);

        // Select kernel
        Kernel kernel = useDoubleAccum ? kernelDouble : kernelByte;

        // Set kernel arguments
        Cl.SetKernelArg(kernel, 0, bufA);
        Cl.SetKernelArg(kernel, 1, bufB);
        Cl.SetKernelArg(kernel, 2, bufC);
        Cl.SetKernelArg(kernel, 3, M);
        Cl.SetKernelArg(kernel, 4, N);
        Cl.SetKernelArg(kernel, 5, K);
        Cl.SetKernelArg(kernel, 6, mulTableBuffer);

        if (useDoubleAccum)
        {
            Cl.SetKernelArg(kernel, 7, toDoubleTableBuffer);
        }
        else
        {
            Cl.SetKernelArg(kernel, 7, addTableBuffer);
        }

        // Execute kernel
        IntPtr[] globalWorkSize = new IntPtr[] { (IntPtr)M, (IntPtr)N };
        Event clevent;
        error = Cl.EnqueueNDRangeKernel(queue, kernel, 2, null, globalWorkSize, null, 0, null, out clevent);
        if (error != ErrorCode.Success)
        {
            Console.WriteLine($"Failed to enqueue kernel: {error}");
            return;
        }

        // Wait for completion
        Cl.Finish(queue);

        // Read results
        error = Cl.EnqueueReadBuffer(queue, bufC, Bool.True, IntPtr.Zero, (IntPtr)C.Length, C, 0, null, out clevent);
        if (error != ErrorCode.Success)
        {
            Console.WriteLine($"Failed to read buffer: {error}");
        }

        // Cleanup
        Cl.ReleaseMemObject(bufA);
        Cl.ReleaseMemObject(bufB);
        Cl.ReleaseMemObject(bufC);
    }

    public static void Cleanup()
    {
        if (!initialized) return;

        Cl.ReleaseMemObject(mulTableBuffer);
        Cl.ReleaseMemObject(addTableBuffer);
        Cl.ReleaseMemObject(toDoubleTableBuffer);
        Cl.ReleaseKernel(kernelDouble);
        Cl.ReleaseKernel(kernelByte);
        Cl.ReleaseCommandQueue(queue);
        Cl.ReleaseContext(context);

        initialized = false;
    }
}
