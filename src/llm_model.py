import torch
import transformers
import logging
from huggingface_hub import logging as hf_logging
from dotenv import load_dotenv

# logging settings
logging.basicConfig(level=logging.INFO)

# hugging face logging settings
hf_logging.set_verbosity_info()

# Load environment variables
load_dotenv()

# llm model
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B"

pipeline = transformers.pipeline(
    "text-generation", model=LLAMA_MODEL_ID, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto"
)

print(pipeline("Hey how are you doing today?"))