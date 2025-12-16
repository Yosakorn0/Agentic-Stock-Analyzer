"""
Force memory cleanup script - Aggressively frees Python memory
"""
import gc
import sys
import os

def clear_memory():
    """Aggressively clear Python memory"""
    print("[CLEANUP] Starting aggressive memory cleanup...")
    
    # Force multiple garbage collection passes
    for i in range(5):
        collected = gc.collect()
        if collected > 0:
            print(f"   Pass {i+1}: Collected {collected} objects")
    
    # Try to clear PyTorch/CUDA cache if available
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            print("   [OK] CUDA cache cleared")
    except ImportError:
        pass
    except Exception as e:
        print(f"   ⚠️  CUDA cleanup error: {e}")
    
    # Try to clear transformers cache if available
    try:
        import transformers
        # Clear any cached models
        if hasattr(transformers, 'file_utils'):
            pass  # transformers may cache files
    except ImportError:
        pass
    
    # Force Python to release memory back to OS (if possible)
    try:
        import ctypes
        # Windows-specific: try to trim working set
        if sys.platform == 'win32':
            kernel32 = ctypes.windll.kernel32
            process_handle = kernel32.GetCurrentProcess()
            # SetProcessWorkingSetSize with -1, -1 to trim
            kernel32.SetProcessWorkingSetSize(process_handle, -1, -1)
            print("   [OK] Requested OS to trim working set")
    except Exception as e:
        print(f"   ⚠️  OS memory trim error: {e}")
    
    print("[OK] Memory cleanup complete")
    print(f"   Python is using ~{sys.getsizeof(gc.get_objects()) / 1024 / 1024:.2f} MB for object tracking")

if __name__ == "__main__":
    clear_memory()
    print("\n💡 Note: Python may still show memory usage in Task Manager.")
    print("   This is normal - Python keeps freed memory for reuse.")
    print("   To fully free RAM, close this Python process.")

