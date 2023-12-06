from typing import List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    content: str
    role: str


class PromptInput(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = Field(
        default=64,
        description="The maximum number of tokens to generate.",
    )

    repeat_penalty: Optional[float] = Field(
        default=1.1,
        description="The penalty to apply to repeated tokens.",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Seed value for the model. If not specified, a random seed will be used.",
    )
