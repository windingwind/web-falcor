// End-to-end exercise kernel for the PixelDebug host tests: prints each
// supported type for the selected pixel and asserts on exactly one pixel.
import Utils.Debug.PixelDebug;

[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId: SV_DispatchThreadID)
{
    const uint2 pixel = dispatchThreadId.xy;
    printSetPixel(pixel);
    print("x", pixel.x);
    print("half", float(pixel.x) * 0.5f);
    print("neg", -int(pixel.y) - 1);
    print("even", (pixel.x & 1) == 0);
    print("vec", float3(1.5f, 2.5f, 3.5f));
    assert(!(pixel.x == 3 && pixel.y == 5), "hit");
}
