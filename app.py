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
    value = {
        "driver_tip": data.driver_tip,
    "distance": data.distance,
    "num_passengers": data.num_passengers,
    "trip_duration": data.trip_duration,
    "payment_method": data.payment_method,
    "rate_code": data.rate_code,
    "extra_charges": data.extra_charges,
    "toll_amount": data.toll_amount
            }
    print(value)
    result = manager(value,load_model())
    return {"prediction":round(result,2)}



if __name__ == "__main__":
    uvicorn.run(app,port=8080)