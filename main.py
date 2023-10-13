# main.py

from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from llm import get_bot_response

app = FastAPI()
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chatbot_response")
async def chatbot_response(user_message: str):
    bot_response = get_bot_response(user_message)
    return {"bot_response": bot_response}

