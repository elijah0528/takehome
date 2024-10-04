from pathlib import Path
from openai import OpenAI
from typing import Optional
from dotenv import load_dotenv
import numpy as np
import difflib
import os

load_dotenv()

OAI_API_KEY= os.getenv("OPENAI_API_KEY")

# This is an interesting experiment that noises text by converting it to speech and then back to text repeatedly to test how the model changes the text.
input = ["ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EXAQUAIZIA, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T.",
         "Seven seashells are silly shells to see."
         ]
class Benchmark:
    def __init__ (self, input_arr):
        client = OpenAI(api_key=OAI_API_KEY)
        self.client = client
        self.input_arr = input_arr
        self.speech_file_path = Path(__file__).parent / "audio_files"
        self.speech_file_path.mkdir(parents=True, exist_ok=True)
        self.proper_nouns = []

    def transcribe_audio(self, speech_file_path, input, changed_words=None):
        audio_file=open(speech_file_path, "rb")
        transcription = self.client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio_file,
            # Prompt is limited to 240 tokens
            # prompt=input,
            response_format="verbose_json",
            timestamp_granularities=["word"]
        )
        print(f"{transcription}\n\n")
        temp_input = transcription.text

        system_prompt = "You are a helpful assistant for the company ZyntriQix. Your task is to correct any spelling discrepancies in the transcribed text. Make sure that the names of the following products are spelled correctly: ZyntriQix, Digique Plus, CynapseFive, VortiQore V8, EchoNix Array, OrbitalLink Seven, DigiFractal Matrix, PULSE, RAPT, B.R.I.C.K., Q.U.A.R.T.Z., F.L.I.N.T. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided. Always return the corrected text and nothing else"

        def generate_corrected_transcript(temperature, system_prompt):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": temp_input
                    }
                ]
            )
            return response.choices[0].message.content
    
        

        result = generate_corrected_transcript(0, system_prompt)

        original_words = input.split()
        corrected_words = result.split()
        print(f"Original: {original_words}")
        print(f"Corrected: {corrected_words}")
        
        diff = difflib.ndiff(original_words, corrected_words)
        for word in diff:
            if word.startswith('- '):
                changed_words.append(word[2:])
        print(f"Changed words: {changed_words}\n")
        for index, word in enumerate(diff):
            if word.startswith('- '):
                removed_word = word[2:]
                original_index = original_words.index(removed_word)

                # Get the words before and after the removed word
                before_word = original_words[original_index - 1] if original_index > 0 else None
                after_word = original_words[original_index + 1] if original_index < len(original_words) - 1 else None

                # Find timestamps for the removed word and its neighbors
                timestamps = []
                for transcription_word in transcription.words:
                    if transcription_word.word in [removed_word, before_word, after_word]:
                        timestamps.append((transcription_word.word, transcription_word.start, transcription_word.end))

                changed_words.append((removed_word, before_word, after_word, timestamps))  # Store the words and their timestamps
        print(f"Changed words with context and timestamps: {changed_words}\n")
        

        return result
        
    def test (self, function, iterations=5, prompt=None):
        results = []
        for index, input in enumerate(self.input_arr):
            transcriptions = []
            temp_input = input
            for i in range(iterations):
                print(f"Iteration {i} ******************")
                response = self.client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=temp_input
                )

                speech_file_loc = self.speech_file_path / f"speech_{index}_{i}.mp3"
                response.stream_to_file(speech_file_loc)
                changed_words = []
                result = function(speech_file_loc, input, changed_words)

                temp_input = result
                transcriptions.append(temp_input)
                print(f"{temp_input}\n")


            results.append((input, transcriptions))
        self.results = results
        return results
    
        
    def evaluate(self, results):
        evaluation_results = []
        for input, transcriptions in results:
            wers = []
            print(f"{input}\n")
            for transcription in transcriptions:
                wer = self.calculate_wer(input, transcription)
                wers.append(wer)
                print(f"WER: {wer} | {transcription}")
            evaluation_results.append((input, wers))

        return evaluation_results

    def calculate_wer(self, reference, hypothesis):
        ref_words = reference.split()
        hyp_words = hypothesis.split()
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

benchmark = Benchmark(input)   
benchmark.test(benchmark.transcribe_audio, 5, input)
benchmark.evaluate(benchmark.results)