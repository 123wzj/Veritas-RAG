import os

from dotenv import load_dotenv

load_dotenv(override=True)

DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')

DEEPSEEK_BASE_URL = os.getenv('DEEPSEEK_BASE_URL')
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
QWEN_API_KEY  =os.getenv("QWEN_API_KEY")
QWEN_BASE_URL = os.getenv("QWEN_BASE_URL")

MINERU_API_KEY = os.getenv("MINERU_API_KEY")
