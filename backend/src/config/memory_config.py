from typing import Dict, Any

MEMORY_CONFIG: Dict[str, Any] = {
    "k": 5,  # Number of messages to keep in memory
    "memory_key": "chat_history",
    "return_messages": True,
    "output_key": "output",
    "input_key": "input"
} 