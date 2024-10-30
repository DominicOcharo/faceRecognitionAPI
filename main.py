from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import face_recognition

app = FastAPI()

# CORS Middleware settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to the specific origins you want to allow
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(face_recognition.router)
