from fastapi import APIRouter
from fastapi import Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

templates = Jinja2Templates(directory="templates")
general_pages_router = APIRouter()


@general_pages_router.get("/")
async def home(request: Request):
	return templates.TemplateResponse("general_pages/home.html",{"request":request})

@general_pages_router.get("/userinput")
async def generateOne(request: Request):
	return templates.TemplateResponse("general_pages/generate-1.html",{"request":request})

@general_pages_router.get("/predict/{cnt}")
async def generateOne(request: Request, cnt:int):
	print('new_page' + str(cnt))
	return templates.TemplateResponse("general_pages/generate-2.html",{"request":request, 'cnt':cnt})

@general_pages_router.get("/result")
async def generateOne(request: Request):
	return templates.TemplateResponse("general_pages/generate-result.html",{"request":request})

@general_pages_router.get("/about")
async def aboutPopup(request: Request):
	return templates.TemplateResponse("general_pages/about.html",{"request":request})

@general_pages_router.get("/predict/{new_prompt}")
async def aboutPopup(request: Request, new_prompt:str):
	return templates.TemplateResponse("general_pages/about.html",{"request":request, 'prompr':new_prompt})

