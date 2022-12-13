import torch
import json 
from fastapi import Body, FastAPI
from fastapi.staticfiles import StaticFiles
import config
from model import FairytaleGenerator
from predict import prediction
from fastapi import Form,Request
from fastapi.encoders import jsonable_encoder

from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from route_homepage import general_pages_router
from config import settings
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel
from flask import request
from typing import Optional
import starlette.status as status
from ast import literal_eval


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

#model_path = 'kogpt2_key(big)/pytorch_model.bin'
model_path = 'trinity_emb_key(small)/pytorch_model.bin'
model = FairytaleGenerator()
model=torch.load(model_path)

tokenizer = config.TOKENIZER
max_len = config.MAX_LEN


'''
@app.get("/")
def ping():
    return {"message": "pong!"}
'''

templates = Jinja2Templates(directory="templates")

def include_router(app):
	app.include_router(general_pages_router)

def configure_static(app):
    app.mount("/static", StaticFiles(directory="static"), name="static")

def start_application():
    app = FastAPI(title=settings.APP_NAME)
    include_router(app)
    configure_static(app)
    return app
    
app = start_application()

class Item(BaseModel):
    prompt: Optional[str] = Form(None)
    storytitle: str = Form(None)
    charactername: str = Form(None)
    placename: str = Form(None)
    keyword_1: str = Form(None)
    keyword_2: str = Form(None)
    keyword_3: str = Form(None)
    firstSentence: str = Form(None)

# @app.get('/predict/0', response_class=HTMLResponse)
# def main(request: Request):
#     print('get_success')
#     return templates.TemplateResponse('/general_pages/generate-2.html', {'request': request, 'cnt':'1'})

@app.get('/predict/{cnt}')
async def main(request: Request):
    print('get_success')
    print(request)
    return templates.TemplateResponse('/general_pages/generate-2.html', {'request': request})

# @app.get('predict/{final_story}')
# def regenerate(final_story:str):
# 	prompt = final_story
# 	pred_1, pred_2, pred_3 = prediction(cnt, prompt, model, tokenizer, max_len, device)
# 	return


@app.post('/predict/{cnt}', response_class=HTMLResponse)
async def userinput(
    request: Request, 
    cnt: int,
    prompt: Optional[str] = Form(None),
    storytitle: str = Form(None),
    charactername: str = Form(None),
    placename: str = Form(None),
    keyword_1: str = Form(None),
    keyword_2: str = Form(None),
    keyword_3: str = Form(None),
    firstSentence: str = Form(None),
    generatedText_1: str = Form(None),
    generatedText_2: str = Form(None),
    generatedText_3: str = Form(None)
    ):
    #cnt = 0
    if int(cnt) < 1:
        keyword = {keyword_1, keyword_2, keyword_3}
        print('++++++++++++'+ str(firstSentence))
        input = {'title':storytitle, 'name':charactername, 'place':placename, 'keyword':keyword, 'firstSentence':firstSentence}
        prompt = f'제목:{input["title"]}/n키워드:{input["name"]}, {input["place"]}, {keyword_1}, {keyword_2}, {keyword_3}/n동화: {input["firstSentence"]}'
        
        print(prompt)

        pred1, pred2, pred3 = prediction(cnt, prompt.strip('<unk>'), model, tokenizer, max_len, device)

        generatedText_1 = str(pred1).strip('<unk>')
        generatedText_2 = str(pred2).strip('<unk>')
        generatedText_3 = str(pred3).strip('<unk>')

        print(generatedText_1)
        print(generatedText_2)
        print(generatedText_3)

        generatedText_1 = " ".join((generatedText_1.split(str(firstSentence)))[0:])
        generatedText_2 = " ".join((generatedText_2.split(str(firstSentence)))[0:])
        generatedText_3 = " ".join((generatedText_3.split(str(firstSentence)))[0:])

        # generatedText_1 = generatedText_1.strip(prompt)
        # generatedText_2 = generatedText_2.strip(prompt)
        # generatedText_3 = generatedText_3.strip(prompt)
       
        print("new!!" + generatedText_1)
        print("new!!" + generatedText_2)
        print("new!!" + generatedText_3)

        return templates.TemplateResponse('/general_pages/generate-2.html', {'request': request,'title':storytitle, 'name':charactername, 
    'place':placename, 'keyword':{keyword_1, keyword_2, keyword_3}, 'final_story':firstSentence, 'input':input, 'prompt':prompt, 'generatedText_1':generatedText_1, 'generatedText_2':generatedText_2, 'generatedText_3':generatedText_3, 'cnt':1})
    else:
        output = await request.json()
        output = jsonable_encoder(output)
        result = json.dumps(output, ensure_ascii=False, sort_keys=True)#.encode('utf-8')
        #print('result: ' + result)
        result = literal_eval(result)
        prompt = result['story']
        prompt = prompt.strip('<br>')
        prompt_list = prompt.split('.')
        new_list_1 = []
        new_list_2 = []

        prompt_list = [new_list_1.append(y) for x in prompt_list for y in x.split('?')]
        prompt_list = [new_list_2.append(y) for x in new_list_1 for y in x.split('!')]
        prompt_list = new_list_2

        print(prompt_list)

        print("*********count : ", cnt)

        if int(cnt) < 3:
            prompt = f'동화:{prompt}'
        else:
            #print(prompt_list)
            prompt = f'동화:{prompt_list[-5]}{prompt_list[-4]}{prompt_list[-3]}{prompt_list[-2]}{prompt_list[-1]}'
        print(type(prompt), prompt)
        print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@come!!!!!!!!!!!!!!!!!!!!!!!")
        cnt = int(result['cnt'])

        storytitle = result['title']
        charactername = result['character']
        placename = result['place']
        keyword = result['keyword']
        input = {'title':storytitle, 'name':charactername, 'place':placename, 'keyword':keyword}
        print('prompt : ' + prompt)

        pred1, pred2, pred3 = prediction(cnt,prompt, model, tokenizer, max_len, device)

        generatedText_1 = str(pred1).strip('<unk>')
        generatedText_2 = str(pred2).strip('<unk>')
        generatedText_3 = str(pred3).strip('<unk>')

        print(generatedText_1)
        print(generatedText_2)
        print(generatedText_3)

        generatedText_1 = " ".join((generatedText_1.split(str(firstSentence)))[0:])
        generatedText_2 = " ".join((generatedText_2.split(str(firstSentence)))[0:])
        generatedText_3 = " ".join((generatedText_3.split(str(firstSentence)))[0:])

        # generatedText_1 = generatedText_1.strip(prompt)
        # generatedText_2 = generatedText_2.strip(prompt)
        # generatedText_3 = generatedText_3.strip(prompt)
       
        print("new!!" + generatedText_1)
        print("new!!" + generatedText_2)
        print("new!!" + generatedText_3)

        result_resposse = {}
        result_resposse.update({"result": "hello"})
        res = {
                'title':storytitle, 
                'name':charactername, 
                'place':placename,
                'keyword':{keyword_1, keyword_2, keyword_3},
                'input':input,
                'prompt':prompt,
                'generatedText_1':generatedText_1,
                'generatedText_2':generatedText_2,
                'generatedText_3':generatedText_3,
                'cnt':cnt
        }
        content = jsonable_encoder(res)
        return JSONResponse(content=content)


    




'''
@app.post('/predict')
async def userinput(
    request: Request, 
    prompt: Optional[str] = Form(None),
    storytitle: str = Form(),
    charactername: str = Form(),
    placename: str = Form(),
    keyword_1: str = Form(),
    keyword_2: str = Form(),
    keyword_3: str = Form()):
    cnt = int(cnt) + 1
    print(type(cnt))
    input = {'title':storytitle, 'name':charactername, 'place':placename, 'keyword':{keyword_1, keyword_2, keyword_3}}
    print(input)
    if cnt < 2:
        prompt = f'제목:{input["title"]}/ 키워드:{input["name"]}, {input["place"]}, {keyword_1}, {keyword_2}, {keyword_3}/ 동화: '
    else: 
        prompt = 'DFDFDFDFDF'
    print(prompt)

    pred1, pred2, pred3 = prediction(prompt, model, tokenizer, max_len, device)
    print(str(pred1))
    print(str(pred2))
    print(str(pred3))
    return templates.TemplateResponse('/general_pages/generate-2.html', {'request': request,'title':storytitle, 'name':charactername, 
    'place':placename, 'keyword':{keyword_1, keyword_2, keyword_3},'input':input, 'prompt':prompt, 'generatedText_1':str(pred1), 'generatedText_2':str(pred2), 'generatedText_3':str(pred3), 'cnt':cnt})
'''

@app.get('/userinput', response_class=HTMLResponse)
def main(request: Request):
    print('useinput GET!')
    return templates.TemplateResponse('/general_pages/generate-1.html', {'request': request})

'''
@app.get('/userinput')
async def inputText(fairytaleTitle:str=Form(...)):
    print(123)
    context = {"title":fairytaleTitle}
    print(context)
    return context


async def inputText(fairytaleTitle:str=Form(...), characterName:str=Form(), placeName:str=Form(), keyword_1:str=Form(), keyword_2:str=Form(), keyword_3:str=Form() ):
    print(fairytaleTitle)
    sentence = str('제목: ') + fairytaleTitle + str('/키워드:') + characterName + str(", ") + placeName + str(", ") + keyword_1 + str(", ") + keyword_2 + str(", ") + keyword_3
    return {"title":fairytaleTitle, "character": characterName, "place":placeName, "keywords": str(keyword_1,keyword_2,keyword_3), 'input':str(sentence)}
'''



'''
@app.post('/userinput', response_class=HTMLResponse)
async def create_input(item: Input):
    global dicted_item
    dicted_item = dict(item)
    dicted_item['success'] = True
    print(dicted_item)
    return JSONResponse(dicted_item)

@app.get("/predict/{sentence}")
def predict(sentence: str):
    pred = prediction(sentence, model, tokenizer, max_len, device)
    return {"message": str(pred)}

@app.get("/predict2")
async def predict():
    print(4,5,6)
    #dicted_item = {k:v for k,v in dict().items()}
    print(1,2,3)
    #dicted_item['success'] = True
    #pred = await prediction(input, model, tokenizer, max_len, device)
    return {"message": "김찬수천재"}
'''
