from typing import Dict, Set
from uuid import uuid4
from datetime import datetime

STUDENTS: Dict[str, dict] = {}   # studentId -> info
SESSIONS: Dict[str, dict] = {}   # sessionId -> session info

def create_session(class_name: str) -> dict:
    session_id = str(uuid4())
    SESSIONS[session_id] = {
        "sessionId": session_id,
        "className": class_name,
        "createdAt": datetime.utcnow().isoformat(),
        "presentIds": set(),  # type: Set[str]
    }
    return {"sessionId": session_id, "className": class_name}
