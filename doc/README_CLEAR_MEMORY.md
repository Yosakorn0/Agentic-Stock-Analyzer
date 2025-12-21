# Clear Memory Documentation

## 📁 File

- **`clear_memory.py`** - Aggressive memory cleanup utility

## 🎯 Purpose

The `clear_memory.py` script provides aggressive memory cleanup for Python processes, especially useful when:
- Running multiple AI model analyses
- Using Hugging Face models that consume significant RAM
- Processing large datasets
- Running long-running scripts that accumulate memory

## 🚀 Usage

### Basic Usage

```bash
python clear_memory.py
```

### Programmatic Usage

```python
from clear_memory import clear_memory

# Run aggressive memory cleanup
clear_memory()
```

## 🔧 What It Does

The script performs multiple memory cleanup operations:

### 1. Garbage Collection

- Runs **5 passes** of Python's garbage collector
- Collects and frees unreferenced objects
- Reports number of objects collected per pass

### 2. CUDA Cache Clearing (if available)

- Clears PyTorch CUDA cache (if CUDA is available)
- Synchronizes CUDA operations
- Runs multiple cache clearing passes

### 3. Transformers Cache Management

- Attempts to clear transformers library cache
- Helps free memory used by cached models

### 4. OS Memory Trimming (Windows)

- Requests OS to trim working set (Windows-specific)
- Uses `SetProcessWorkingSetSize` API
- Helps return memory to the operating system

## 📊 Output

```
[CLEANUP] Starting aggressive memory cleanup...
   Pass 1: Collected 1234 objects
   Pass 2: Collected 567 objects
   Pass 3: Collected 89 objects
   Pass 4: Collected 12 objects
   Pass 5: Collected 0 objects
   [OK] CUDA cache cleared
   [OK] Requested OS to trim working set
[OK] Memory cleanup complete
   Python is using ~45.23 MB for object tracking
```

## ⚙️ When to Use

### Recommended Use Cases

1. **After running multiple AI analyses**
   ```python
   # After scanning many stocks with AI
   from clear_memory import clear_memory
   clear_memory()
   ```

2. **Before loading large models**
   ```python
   # Free memory before loading Hugging Face models
   clear_memory()
   # Then load model
   ```

3. **Between batch operations**
   ```python
   # Between processing batches
   for batch in batches:
       process_batch(batch)
       clear_memory()  # Clean up after each batch
   ```

4. **After backtesting**
   ```python
   # After running backtests with large datasets
   results = run_backtest()
   clear_memory()
   ```

### Not Recommended For

- **Frequent calls** (e.g., every second) - Overhead may outweigh benefits
- **During critical operations** - May cause brief pauses
- **When memory is already low** - May not help significantly

## 🔍 Memory Management Notes

### Python Memory Behavior

- **Python keeps freed memory**: Python's memory allocator keeps freed memory for reuse
- **OS may not show freed RAM**: Task Manager may still show memory usage
- **This is normal**: Python optimizes for performance by reusing memory

### What This Script Does

- **Frees Python objects**: Removes references and collects garbage
- **Clears GPU memory**: Frees CUDA cache if available
- **Requests OS trim**: Asks OS to reclaim memory (Windows)

### What It Doesn't Do

- **Doesn't guarantee OS memory release**: OS decides when to reclaim
- **Doesn't close Python process**: To fully free RAM, close Python
- **Doesn't prevent memory leaks**: Fix code issues, not symptoms

## 💡 Best Practices

### 1. Use After Heavy Operations

```python
# Heavy operation
results = multi_ai_scanner.scan_stocks(...)

# Clean up
clear_memory()
```

### 2. Use Before Loading Large Models

```python
# Free memory first
clear_memory()

# Then load model
model = load_huggingface_model(...)
```

### 3. Monitor Memory Usage

```python
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")

clear_memory()

print(f"Memory after cleanup: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

## ⚠️ Limitations

1. **OS Memory**: Python may not release memory back to OS immediately
2. **GPU Memory**: CUDA cache clearing requires PyTorch and CUDA
3. **Windows Only**: OS memory trimming is Windows-specific
4. **Not a Fix**: Doesn't fix memory leaks in your code

## 🔧 Integration Examples

### With Multi-AI Scanner

```python
from multi_ai_scanner import MultiAIScanner
from clear_memory import clear_memory

scanner = MultiAIScanner()

# Scan stocks
results = scanner.scan_stocks(...)

# Clean up after scanning
clear_memory()
```

### With Gold Scalping

```python
from gold_scalping_live import monitor_gold_scalping
from clear_memory import clear_memory
import signal

def cleanup_on_exit(signum, frame):
    clear_memory()
    exit(0)

signal.signal(signal.SIGINT, cleanup_on_exit)

# Monitor gold
monitor_gold_scalping()
```

### With Batch Processing

```python
from clear_memory import clear_memory

stocks = ['AAPL', 'MSFT', 'GOOGL', ...]

for i, stock in enumerate(stocks):
    analyze_stock(stock)
    
    # Clean up every 10 stocks
    if (i + 1) % 10 == 0:
        clear_memory()
        print(f"Cleaned memory after {i + 1} stocks")
```

## 📚 Related Files

- **`multi_ai_scanner.py`** - Uses multiple AI models (high memory usage)
- **`core/analysis/huggingface_analyzer.py`** - Hugging Face model integration
- **`gold_scalping_live.py`** - Long-running monitoring script

## 💡 Tips

1. **Run after heavy operations** - Not during critical sections
2. **Monitor memory before/after** - Verify it's helping
3. **Don't overuse** - Too frequent calls add overhead
4. **Fix root causes** - Address memory leaks in code
5. **Use with long-running scripts** - Helps prevent gradual memory growth

---

**Note**: This script helps manage memory but doesn't fix underlying memory leaks. Always profile and fix memory issues in your code!

