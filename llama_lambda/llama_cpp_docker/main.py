import logging
import os
import random

from fastapi import FastAPI
from mangum import Mangum

from schemas import PromptInput
from utils import MessageFormatter


logger = logging.getLogger(__name__)

MODELPATH = "/opt/modelfile.gguf"
stage = os.environ.get('STAGE', None)
openapi_prefix = f"/{stage}" if stage else "/"
app = FastAPI(
    title="OpenLLaMa on Lambda API",
    openapi_prefix=openapi_prefix,
)
llm = None

@app.post("/prompt")
async def prompt(
    item: PromptInput,
):
    global llm

    from llama_cpp import Llama

    seed = item.seed
    if item.seed is None: 
        seed = random.randint(0, 65535)

    if llm is None:
        llm = Llama(model_path=MODELPATH, seed=seed)

    formatter = MessageFormatter("chat_ml")
    output = llm(
        formatter.to_instruct(item.messages, with_response_suffix=True),
        repeat_penalty=item.repeat_penalty,
        echo=False,
        max_tokens=item.max_tokens,
    )

    return output


handler = Mangum(app)
