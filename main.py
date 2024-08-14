from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llamaModel import llama3


llama = llama3.Llama3()

app = FastAPI()

# CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/llama")
def llama_model(user_prompt: str):
    return llama.run(user_prompt)

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}