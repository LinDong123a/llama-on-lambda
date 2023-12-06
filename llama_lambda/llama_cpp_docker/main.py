import logging
import os
import random

from fastapi import FastAPI
from mangum import Mangum

from schemas import PromptInput


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

    if item.seedval is None: 
        seedval = random.randint(0, 65535)

    if llm is None:
        llm = Llama(model_path=MODELPATH, seed=seedval)

    output = llm(
        " Below is an instruction that describes a task, as well as any previous text you have generated. You must continue where you left off if there is text following Previous Output. Write a response that appropriately completes the request. When you are finished, write [[COMPLETE]].\n\n Instruction: "+text+" Previous output: "+prioroutput+" Response:",
        repeat_penalty=item.repeat_penalty,
        echo=False,
        max_tokens=item.max_tokens,
    )

    return output


handler = Mangum(app)
