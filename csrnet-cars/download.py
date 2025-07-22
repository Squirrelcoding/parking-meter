import os

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv(".env.local")

rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
# project = rf.workspace("elpida-eleftheriadi").project("carpk-xk8e1")
project = rf.workspace("data-y0pgy").project("cowc-llfne-neo6m")
version = project.version(1)
dataset = version.download("coco")
