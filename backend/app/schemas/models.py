from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str
    role: str = "user"


class UserLogin(BaseModel):
    username: str
    password: str


class Review(BaseModel):
    text: str


class FeedbackPayload(BaseModel):
    timestamp: str
    feedback: str  # "correct" | "incorrect"
