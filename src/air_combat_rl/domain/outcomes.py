from __future__ import annotations
from enum import Enum

class Outcome(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    HIT = "hit"
    CRASH = "crash"
    EXHAUSTED = "exhausted"
    TIMEOUT = "timeout"
