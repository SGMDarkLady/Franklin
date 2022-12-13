import config
import torch
from transformers import GPT2LMHeadModel

import numpy as np
import torch
import easydict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
import json
import torch


from model import FairytaleGenerator

def prediction(cnt, text, _model, _tokenizer, _max_len, _device):
    
    text_final_1 = ''
    text_final_2 = ''
    text_final_3 = ''

    args=easydict.EasyDict({
        "model_type":None,    
        "model_name_or_path":None,
        "prompt":"",
        "length":100,
        "stop_token":"",
        "temperature":0.9,
        "repetition_penalty":4.0,
        "k":0,
        "p":0.9,
        "padding_text":"",
        "xlm_language":"",
        "seed":100,
        #         "no_cuda":"store_true",
        "use_auth_token":True,
        "device":"cuda",
        "n_gpu":0
    })
    #_model.to(args.device)

    #prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    
    # 이 부분 필요한지 확인 -> 불필요하다면 위 함수들 싹 지워도 됨
    # Different models need different input formatting and/or extra arguments

    encoded_prompt = _tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    encoded_prompt = encoded_prompt.to(args.device)
    _model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH).to(device='cuda', non_blocking=True)

    output_sequences = _model.generate(
        input_ids=encoded_prompt,
        max_length=350,
        attention_mask=None,
        temperature=args.temperature,
        top_k=args.k,
        top_p=args.p,
        repetition_penalty=args.repetition_penalty,
        do_sample=True,
        pad_token_id=_tokenizer.eos_token_id,
        return_dict=True,   
        num_return_sequences=3,
    )

    generated_sequence_1 = output_sequences[0].tolist()
    generated_sequence_2 = output_sequences[1].tolist()
    generated_sequence_3 = output_sequences[2].tolist()

    result_text_1 = _tokenizer.decode(generated_sequence_1, clean_up_tokenization_spaces=True)
    result_text_2 = _tokenizer.decode(generated_sequence_2, clean_up_tokenization_spaces=True)
    result_text_3 = _tokenizer.decode(generated_sequence_3, clean_up_tokenization_spaces=True)

    # text_final_1 += parse_text(result_text_1.split('동화:')[1]) + '.'
    # text_final_2 += parse_text(result_text_2.split('동화:')[1]) + '.'
    # text_final_3 += parse_text(result_text_3.split('동화:')[1]) + '.'

    #print("********" + type(result_text_1.split('동화:')[1]))
    r1 = (result_text_1.split('동화:')[1]).split('.')
    r2 = (result_text_2.split('동화:')[1]).split('.')
    r3 = (result_text_3.split('동화:')[1]).split('.')

    count = 3
    if cnt < 2:
        count = 0
    text_final_1 += (r1[count] + ". " + r1[count+1]+ ".")
    text_final_2 += (r2[count] + ". " + r2[count+1] + ".")
    text_final_3 += (r3[count] + ". " + r3[count+1] + ".")

    return text_final_1, text_final_2, text_final_3

if __name__ == "__main__":

    if torch.cuda.is_available():
        _device = torch.device("cuda")
    else:
        _device = torch.device("cpu")

    _model = GPT2LMHeadModel.from_pretrained(config.MODEL_PATH)
    _model.load_state_dict(torch.load(config.MODEL_PATH))

    _model.to(_device)
    _model.eval()
    _tokenizer = config.TOKENIZER
    _max_len = config.MAX_LEN
    #_num_return_sequences = config.NUM_RETURN_SEQUENCES
    _prompt = str("안녕하세요 저는 ")
    pos = prediction(_prompt, _model, _tokenizer, _max_len, _device)

    print('prediction : ' + pos)

def parse_text(text):
    stop_token = '.'
    result_text = ''
    text = text.replace(text,"").strip()
    text = text[: text.find(stop_token) if stop_token else None]
    result_text += text+'.'
    return result_text