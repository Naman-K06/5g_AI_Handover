"""
5G AI Handover Project - Main Module
Contains physics engine, configuration, and utilities for 3GPP handover optimization
"""

__version__ = "1.0.0"
__author__ = "5G AI Research Team"

from .config import *
from .physics_engine import get_network_state, get_5g_throughput

__all__ = ['get_network_state', 'get_5g_throughput']
