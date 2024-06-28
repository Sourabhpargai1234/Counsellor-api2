import os
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from gradientai import Gradient
from gradientai.openapi.client.exceptions import UnauthorizedException
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# Load environment variables from .env file
load_dotenv()

GRADIENT_ACCESS_TOKEN = os.getenv('GRADIENT_ACCESS_TOKEN')
GRADIENT_WORKSPACE_ID = os.getenv('GRADIENT_WORKSPACE_ID')

app = FastAPI()

# Allow CORS for local development and testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the model once and reuse it
gradient = Gradient(access_token=GRADIENT_ACCESS_TOKEN)
base_model = gradient.get_base_model(base_model_slug="nous-hermes2")
new_model_adapter = base_model.create_model_adapter(name="test model 3")

class Question(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "Server is running successfully"}


@app.post("/ask")
async def ask_question(question: Question):
    sample_query = f"### Instruction: {question.question} \n\n### Response:"

    try:
        print(f"Asking: {sample_query}")
        
        # Use asyncio to handle the request asynchronously
        completion = await asyncio.to_thread(
            new_model_adapter.complete, query=sample_query, max_generated_token_count=100
        )
        response = completion.generated_output
        print(f"Generated (before fine-tune): {response}")

        return JSONResponse(content={"response": response})

    except UnauthorizedException as e:
        print("Unauthorized: Check your API key and permissions.")
        raise HTTPException(status_code=401, detail="Unauthorized: Check your API key and permissions.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
