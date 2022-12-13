from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    GPT2Config,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
    PreTrainedTokenizerFast,
    BartForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM
)

from pydantic import BaseSettings

TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 16
EPOCHS = 50
#MODEL_PATH = "kogpt2_key(big)"
MODEL_PATH = "trinity_emb_key(small)"
MAX_LEN = 100
#TOKENIZER  = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2")
TOKENIZER  = AutoTokenizer.from_pretrained("skt/ko-gpt-trinity-1.2B-v0.5")
NUM_RETURN_SEQUENCES = 3

class Settings(BaseSettings):
    APP_NAME: str = "Franklin"

settings = Settings()