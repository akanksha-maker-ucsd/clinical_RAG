import streamlit as st
import os
import faiss
import torch
torch._classes = types.SimpleNamespace()
sys.modules['torch._classes'] = torch._classes
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from together import Together
from typing import List, Dict, Tuple
import re
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import pandas as pd
import io
os.environ["TOGETHER_API_KEY"] = st.secrets["TOGETHER_API_KEY"]
# os.environ["HUGGINGFACE_HUB_TOKEN"] = st.secrets["HUGGINGFACE_HUB_TOKEN"]
# from huggingface_hub import login
# login(token=os.environ["HUGGINGFACE_HUB_TOKEN"])
