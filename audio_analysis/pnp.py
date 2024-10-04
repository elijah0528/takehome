from pathlib import Path
from openai import OpenAI
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import numpy as np
import difflib
import os
import datetime
import json
import time
import keyboard as kbd
import re

load_dotenv()

OAI_API_KEY= os.getenv("OPENAI_API_KEY")

# This is an interesting experiment that noises text by converting it to speech and then back to text repeatedly to test how the model changes the text.
input_text = ["ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EXAQUAIZIA, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.",
         "Seven seashells are silly shells to see.",
         "Seven seashells are silly shells to see.",

         "Artificial intelligence is transforming the world.",
         "Natural language processing enables machines to understand human language.",
         "Deep learning models require large datasets for training.",
         "The quick brown fox jumps over the lazy dog.",

]

class Hard_Word(BaseModel):
    word: str
    attempted_word: str
    before_word: Optional[str]
    after_word: Optional[str]
    timestamps: Optional[tuple[float, float]]
    snippet_url: str

class PNP:
    def __init__ (self, input_text):
        client = OpenAI(api_key=OAI_API_KEY)
        self.client = client
        self.input_text = input_text
        self.system_prompt = ""

        # Defines where to save audio files
        self.speech_file_path = Path(__file__).parent / "audio_files"
        self.speech_file_path.mkdir(parents=True, exist_ok=True)

        # Save difficult words – usually proper nouns
        self.hard_word_array = []
        self.hard_words = []
        
        self.hard_words_data = self.update_corrections_dictionary()

    def update_corrections_dictionary (self):
        with open("./hard_words.json", "r") as json_file:
            data = json.load(json_file)
            return data 
        
    # Speak the text
    def generate_audio(self, text, index, iteration): # Returns the location of the audio file
        response = self.client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        timestamp = datetime.datetime.now().strftime("%H%M%S")
        speech_file_loc = self.speech_file_path / f"speech_{timestamp}_{''.join(text.split()[:3])}.mp3"
        response.stream_to_file(speech_file_loc)
        return speech_file_loc

    # Transcribe the audio  
    def transcribe_audio(self, speech_file_path):  # Returns the transcribed text
        audio_file = open(speech_file_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        return (speech_file_path, transcription)
    
    def type_and_capture_input(self, text):  # Get user input intermediately (optional)
        print("\nMake your edits and press Enter:\n")
        kbd.write(text)
        
        user_input = input()

        return user_input.strip()

    def apply_known_corrections(self, corrected_transcript):  # Simply replace known corrections
        for entry in self.hard_words_data: 
            corrected_transcript = corrected_transcript.replace(entry['attempted_word'], entry['word'])
        return corrected_transcript
    
    # GPT-4o to correct the transcription
    def generate_corrected_transcript(self, input_text, temperature=0): # Returns the transcribed correct text 
        self.system_prompt = f"You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. Always return the corrected text and nothing else."

        if len(self.hard_word_array) > 0:
                self.system_prompt += f" Some common mistakes will be given to you in the form (correct spelling : incorrect spelling). "
        for corr, incorr in self.hard_word_array:
            self.system_prompt += f"({corr} : {incorr}), "
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": input_text
                }
            ]
        )
        corrected_transcript = response.choices[0].message.content

        corrected_transcript = self.apply_known_corrections(corrected_transcript)

        # Allows for human input only if its entered into the terminal
        # Comment out if you don't want to edit the transcript
        corrected_transcript = self.type_and_capture_input(corrected_transcript)
        
        return corrected_transcript

    def diff_correction(self, reference, text): # Returns a tuple of (correct_word, incorrect_transcription)
        ref_words = reference.split()
        text_words = text.split()

        for ref_index, word in enumerate(ref_words):
            if word in text_words:
                text_index = text_words.index(word) 
                text_words[text_index] = "|"
                ref_words[ref_index] = "|"
        ref_hard = []
        text_hard = []

        curr_hard_word = ""
        for word in ref_words:
            if word == "|" and curr_hard_word.strip() == "":
                continue
            if word == "|" and curr_hard_word.strip() != "":
                ref_hard.append(curr_hard_word.strip())
                curr_hard_word = ""
            else:
                curr_hard_word += word + " "
                print(f"Current Hard Word - right: {curr_hard_word}")
        if curr_hard_word.strip():
            ref_hard.append(curr_hard_word.strip())

        curr_hard_word = ""
        for word in text_words:
            if word == "|" and curr_hard_word.strip() == "":
                continue
            elif word == "|" and curr_hard_word.strip() != "":
                text_hard.append(curr_hard_word.strip())
                curr_hard_word = ""
            else:
                curr_hard_word += word + " "
                print(f"Current Hard Word - wrong: {curr_hard_word}")
        if curr_hard_word.strip():
            text_hard.append(curr_hard_word.strip())

        # Return a list of tuples of the form (reference_hard_word, text_hard_word)
        return list(zip(ref_hard, text_hard))

    def display_corrections(self, corr_incorr_pairs): # Displays the corrections in a formatted manner
        # Note this ASSUMES that the len of hard_word_array is greater than 0
        for ref_word, text_word in self.hard_word_array:
            print(f"{ref_word:<60} | {text_word}")
    

    def get_timestamp(self, hard_word, transcription): # Returns the timestamp of the hard word
        hard_words = re.findall(r'\b\w+\b', hard_word)  # Split the hard_word into individual words AND remove punctuation
        start = float('inf')
        end = -float('inf')
        for i, word in enumerate(transcription.words):
            print(f"Word: {word.word}")
            if word.word in hard_words:  # Check if the word is in the group
                start = min(start, word.start)
                end = max(end, word.end)

        if start == float('inf') or end == -float('inf'):
            return None
        return (start, end)  # Return start and end timestamps as a tuple
            
    def build_hard_words(self, corr_incorr_pairs, transcription, file_path): # Builds the hard words list under the Hard_Word format
        words = transcription.text.split()  # Split the transcription into words
        for ref_word, text_word in corr_incorr_pairs:
            timestamps = self.get_timestamp(text_word, transcription)
            text_word_array = text_word.split()
            start = text_word_array[0]
            end = text_word_array[-1]
            # Find the index of the attempted word in the transcription
            start_index = words.index(start)
            end_index = words.index(end)
            
            # Get the words before and after the attempted word
            before_word = words[start_index - 1] if start_index > 0 else ""
            after_word = words[end_index + 1] if end_index < len(words) - 1 else ""


            hard_word = Hard_Word(word=ref_word, attempted_word=text_word, before_word=before_word, after_word=after_word, timestamps=timestamps, snippet_url=str(file_path))
            self.hard_words.append(hard_word)

    def display_hard_words(self):
        for hard_word in self.hard_words:
            print(f"Word: {hard_word.word:<60}\nAttempted word: {hard_word.attempted_word}\nBefore word: {hard_word.before_word}\nAfter word: {hard_word.after_word}\nTimestamps: {hard_word.timestamps}\nSnippet URL: {hard_word.snippet_url}\n\n")

    # Extend json
    def save_hard_words(self):
        try:
            with open("hard_words.json", "r") as json_file:
                existing_data = json.load(json_file)
        except FileNotFoundError:
            existing_data = []
            
        new_data = []
        for hard_word in self.hard_words:
            new_data.append({
                "word": hard_word.word,
                "attempted_word": hard_word.attempted_word,
                "before_word": hard_word.before_word,
                "after_word": hard_word.after_word,
                "timestamps": hard_word.timestamps,
                "snippet_url": hard_word.snippet_url
            })

        # Combine existing data with new data
        existing_data.extend(new_data)

        # Save the combined data back to the JSON file
        with open("hard_words.json", "w") as json_file:
            json.dump(existing_data, json_file, indent=4) 
    # Calculate the word error rate
    def calculate_wer(self, reference, hypothesis): # Returns the word error rate
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        print(f"Input: {ref_words}")
        print(f"Corrected Transcription: {hyp_words}")
        d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=int)

        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j

        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                if ref_words[i - 1] == hyp_words[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    substitution = d[i - 1][j - 1] + 1
                    insertion = d[i][j - 1] + 1
                    deletion = d[i - 1][j] + 1
                    d[i][j] = min(substitution, insertion, deletion)

        wer = d[len(ref_words)][len(hyp_words)] / len(ref_words)
        return wer   

    def test(self, input_text):
        for i, text in enumerate(input_text):
            speech_file_loc = self.generate_audio(text, i, 0)
            file_path, transcription = self.transcribe_audio(speech_file_loc)
            print(f"Transcription: {transcription.text}\n\n")
            corrected_transcript = self.generate_corrected_transcript(transcription.text)
            # When using learning, reaplce corrected_transcript with corrected transcript after manual edits too
            corr_incorr_pairs = self.diff_correction(corrected_transcript, transcription.text)
            print(f"Corr incorr pairs: {corr_incorr_pairs}")
            self.hard_word_array.extend(corr_incorr_pairs)

            self.build_hard_words(corr_incorr_pairs, transcription, file_path)
            self.display_hard_words()
            self.save_hard_words()
            self.hard_words = []

            self.update_corrections_dictionary()

            wer = self.calculate_wer(text, corrected_transcript)
            print(f"WER: {wer}\n\n")

pnp = PNP(input_text)
pnp.test(input_text)