from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from llamaModel import Llama3


llama = Llama3

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

@app.post("/llama/")
def llama_model(user_prompt: str):
    return llama.run(user_prompt)

