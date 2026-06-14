import torch
from torch.nn import functional as F

def check_sdpa():
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")

    # Check for FlashAttention 2
    try:
        from torch.backends.cuda import flash_sdp_enabled, mem_efficient_sdp_enabled, math_sdp_enabled
        print(f"Flash Attention Enabled: {flash_sdp_enabled()}")
        print(f"Mem Efficient Attention Enabled: {mem_efficient_sdp_enabled()}")
        print(f"Math Attention Enabled: {math_sdp_enabled()}")
    except ImportError:
        print("Could not check SDP backends (older PyTorch?)")

    # Test SDPA with different dtypes
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        print(f"\nTesting SDPA with {dtype}...")
        q = torch.randn(1, 8, 128, 64, device='cuda', dtype=dtype)
        k = torch.randn(1, 8, 128, 64, device='cuda', dtype=dtype)
        v = torch.randn(1, 8, 128, 64, device='cuda', dtype=dtype)
        
        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=False, enable_mem_efficient=False):
                out = F.scaled_dot_product_attention(q, k, v)
                print(f"  FlashAttention 2: SUCCESS")
        except Exception as e:
            print(f"  FlashAttention 2: FAILED ({type(e).__name__})")

        try:
            with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
                out = F.scaled_dot_product_attention(q, k, v)
                print(f"  MemEfficient Attention: SUCCESS")
        except Exception as e:
            print(f"  MemEfficient Attention: FAILED ({type(e).__name__})")

if __name__ == "__main__":
    check_sdpa()
