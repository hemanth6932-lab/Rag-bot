from fastapi import FastAPI
from Api.routes import router

app = FastAPI()

app.include_router(router)
