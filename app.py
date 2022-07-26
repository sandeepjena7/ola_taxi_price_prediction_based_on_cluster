from engine import log,manager,load_model 
import logging
from pydantic import BaseModel
from fastapi import FastAPI ,Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse,FileResponse,JSONResponse
from typing import Dict
import uvicorn

# log("config/logging.yml").setup_logging()
# logger = logging.getLogger("api")

class Input(BaseModel):
    driver_tip: int
    distance: int
    num_passengers: int
    trip_duration: int
    payment_method: int
    rate_code: int
    extra_charges: int
    toll_amount: int

class InputNotValid(Exception):
    def __init__(self,message:str=None):
        self.message = message


app  = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST","GET"], 
    allow_headers=["*"],
    max_age=2 # how mcuh hit api per second
    )



@app.exception_handler(InputNotValid)
def not_valid(request:Request,exc:InputNotValid):
    
    return JSONResponse(
        status_code=418
        ,content = {"message":f"{exc.message} "}
                        )


@app.post("/predict",response_model=Dict[str,float])
def predict(data:Input):
    value = data
    result = manager(value,load_model())
    return {"prediction":result}



if __name__ == "__main__":
    uvicorn.run(app,port=8080)