from collections import namedtuple
from typing import List


Message = namedtuple("Message", ["content", "role"])


class MessageFormatter():
    chat_ml_format = {
        "system": """<|im_start|>system\n{content}<|im_end|>""",
        "user": """<|im_start|>user\n{content}<|im_end|>""",
        "assistant": """<|im_start|>assistant\n{content}<|im_end|>""",
    }

    def __init__(self, format: str) -> None:
        self.formatter = {}
        if format == "chat_ml":
            self.formatter = self.chat_ml_format
        else:
            raise ValueError("Invalid format specified.")
    
    def to_instruct(self, messages: List[Message], with_response_suffix: bool = True):
        """Converts a list of messages into an instruction string.
        
        Args:
            messages (List[Message]): A list of messages to convert.
            with_response_suffix (bool, optional):
                Whether to append a response suffix to the instruction. Defaults to True.
        """
        message_str_list = []
        for message in messages:
            message_str_list.append(
                self.formatter[message.role].format_map({"content": message.content})
            )

        prompt = "\n".join(message_str_list)
        
        if with_response_suffix:
            prompt += "\n<|im_start|>assistant\n"

        return prompt
