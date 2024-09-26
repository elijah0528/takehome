from dotenv import load_dotenv
import os

load_dotenv()

OAI_API_KEY= os.getenv("OPENAI_API_KEY")

print(OAI_API_KEY)