from openai import OpenAI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os

client = OpenAI()

load_dotenv()

OAI_API_KEY= os.getenv("OPENAI_API_KEY")


class Transcript(BaseModel):
    id: str
    text: str
    confidence: Optional[float]


class Listener:
    def __init__ (self):
        self.client = client

    def transcribe (self, audio):
        audio_file= open(audio, "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        """
        response: {
            text: string,
            task: string,
            language: string,
            duration: float,
            words[{
                word: string
                start: float
                end: float
            }]
        }
        """
        return transcription
    

listener = Listener()
text = listener.transcribe("../test_audio.mp3")
print(text)
    

    