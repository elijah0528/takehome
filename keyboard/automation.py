import speech_recognition as sr
import asyncio
import time
from pynput import keyboard
import keyboard as kbd

# Recognizes speech using sr 
def recognize_speech():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = recognizer.listen(source)
    
    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Speech not understood")
        return None

# Defines graceful quitting
def on_press(key):
    try:
        if key.char == 'q':
            print("Quitting...")
            return False
    except AttributeError:
        pass
    
print("Press 'q' to quit the program")

# Starts keyboard listener
listener = keyboard.Listener(on_press=on_press)
listener.start()

# Listens for more speech
while listener.running:
    text = recognize_speech()
    if text:
        kbd.write(text)
    time.sleep(0.1)