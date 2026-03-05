import os
import sys

# Load .env before starting server
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Paddle flags must be set before any paddle/paddleocr import
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import uvicorn
uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_level="info")
