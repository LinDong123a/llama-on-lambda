import logging
import os
import random

from fastapi import FastAPI
from mangum import Mangum

from schemas import PromptInput
from utils import MessageFormatter


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

    logger.info("开始导入llama.cpp")
    from llama_cpp import Llama

    seed = item.seed
    if item.seed is None: 
        seed = random.randint(0, 65535)

    if llm is None:
        logger.info("No model loaded, loading...")
        llm = Llama(model_path=MODELPATH, seed=seed)

    logger.info("Loading Finished")

    formatter = MessageFormatter("chat_ml")
    
    logger.info("Start responding...")
    output = llm(
        formatter.to_instruct(item.messages, with_response_suffix=True),
        repeat_penalty=item.repeat_penalty,
        echo=False,
        max_tokens=item.max_tokens,
    )
    logger.info("Finish reponding")

    return output


handler = Mangum(app)
