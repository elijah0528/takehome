from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import os

client = OpenAI()


class Probable_Error(BaseModel):
    reasoning: str
    error_probability: float



load_dotenv()

OAI_API_KEY= os.getenv("OPENAI_API_KEY")

class Analyzer:
    def __init__(self):
        self.client = client
    
    def analyze(self, text):
        analysis = client.beta.chat.completions.parse(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system",
                    "content": "Analyze the following text and provide the probability that the sentence has a typo in it or is grammatically incorrect. This could include incorrectly spelled proper nouns and such. Only output a probability as a float between 0 and 1 or you lose your job"
                },
                {
                    "role": "user",
                    "content": "I started a company Axios with Mikhail Botvinnik"
                },
                {
                    "role": "assistant",
                    "content": 0.2
                },
                {
                    "role": "user",
                    "content": text
                },
            ],
            response_format=Probable_Error,
        )

        return analysis

    def new_analyzer(self, text):
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                {
                    "role": "system", 
                    "content": """
                    Analyze the following text and provide the probability that the sentence has a typo in it or is 
                    grammatically incorrect. This could include incorrectly spelled proper nouns and such. 
                    You must output the word, the probability it is erroneous and the context in which is spoken. 
                    The context and word must be exactly the same as the text and the probility must be a float between 
                    0 and 1
                    """
                },
                {
                    "role": "user", 
                    "content": text
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "error",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "errors": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "word": {"type": "string"},
                                        "probability": {"type": "number"},
                                        "context":  {"type": "string"}
                                    },
                                    "required": ["word", "probability", "context"],
                                    "additionalProperties": False
                                }
                            },
                        },
                        "required": ["errors"],
                        "additionalProperties": False

                    },
                    "strict": True
                }
            }
        )

        return response.choices[0].message
    
analyzer = Analyzer()
response = analyzer.new_analyzer("I started a company Axios. I started th this with my friend Mika Botvinnik. He is is a great guy!")
print(response)