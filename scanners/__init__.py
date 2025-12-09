"""
Scanner implementations

AgenticStockScanner: Unified scanner supporting both sequential and parallel processing
- Use parallel=True for faster analysis of multiple stocks
- Use parallel=False (default) for sequential processing
"""
from .agentic_scanner import AgenticStockScanner

__all__ = ['AgenticStockScanner']
