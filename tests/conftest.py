"""Shared pytest setup: import speech_bridge from the repository root."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
