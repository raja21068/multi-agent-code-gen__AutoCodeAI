from pydantic import BaseModel

class Conversation(BaseModel):
    id: str
    participants: list[str]
    messages: list[str]
    start_time: str
    end_time: str

class Task(BaseModel):
    id: str
    description: str
    assigned_to: str
    status: str
    created_at: str
    updated_at: str

class Session(BaseModel):
    id: str
    user_id: str
    conversation_id: str
    created_at: str
    updated_at: str
