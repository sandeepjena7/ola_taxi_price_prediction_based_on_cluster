import streamlit as st
import json
import logging
import os
import yaml
from engine import manager,load_model,log


log("config/logging.yml").setup_logging()
logger = logging.getLogger("streamlite")
stream = '---'
logger.info(f"* {stream*10}>")


if not os.path.isfile("report/schema_input.json"):
    logger.error("schema_input.json path not found")
    raise FileNotFoundError("Go to notebook generate schema_input.json and put report folder")
with open("report/schema_input.json",'r') as f:
    schema = json.load(f)

user = {}
data = {}
for key ,item in schema.items():
    user[key] = st.slider(f"{key}",item['min'],item['max'],(item['min']+item['max'])/2)
for key in schema.keys():
    data[key] = int(user[key])

result = manager(data,load_model())
st.write("Total Amount is: ",round(result,2),"₹")
logger.info(f"<{stream*10} *")

