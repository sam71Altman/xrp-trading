from enum import Enum

class BotState(Enum):
    IDLE = 0
    ENTERED = 1        # REQUIRED — Signal accepted
    OPEN = 2           # Order confirmed
    CLOSING = 3
    CLOSED = 4

# Startup Integrity Check
REQUIRED_STATES = ["IDLE", "ENTERED", "OPEN", "CLOSING", "CLOSED"]
for s in REQUIRED_STATES:
    if not hasattr(BotState, s):
        raise RuntimeError(f"CRITICAL: Missing BotState.{s}")
