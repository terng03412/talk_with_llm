import os
import time
import queue
import threading
import numpy as np
import pygame
import sounddevice as sd
import torch
from scipy.io.wavfile import write
from transformers import pipeline
from openai import OpenAI
from langchain_community.llms import Ollama
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory
from dotenv import find_dotenv, load_dotenv

# Load environment variables
load_dotenv(find_dotenv())

# OpenAI setup
client = OpenAI()

# Audio settings
FREQ = 16000
MODEL_NAME = "biodatlab/whisper-th-small-combined"
device = 0 if torch.cuda.is_available() else "cpu"

# Initialize components
pygame.init()
pygame.mixer.init()
pipe = pipeline("automatic-speech-recognition", model=MODEL_NAME, device=device)
llm = Ollama(model="llama3.1")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize agent
agent = initialize_agent(
    [],
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
    max_iterations=2,
    early_stopping_method="generate",
    handle_parsing_errors=True
)

def speak(text):
    for sentence in text.split('\n'):
        if sentence.strip():
            response = client.audio.speech.create(
                model="tts-1", voice="nova", input=sentence
            )
            with open("output.mp3", "wb") as f:
                for chunk in response.iter_bytes(chunk_size=4096):
                    f.write(chunk)
            
            pygame.mixer.music.load("output.mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            time.sleep(0.5)

def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata.copy())

def record_audio():
    with sd.InputStream(samplerate=FREQ, channels=1, callback=audio_callback):
        print("Recording... Press Enter to stop.")
        stop_recording.wait()

def transcribe_audio():
    audio_data = []
    while not q.empty():
        audio_data.append(q.get())
    recording = np.concatenate(audio_data, axis=0)
    write("recording.wav", FREQ, recording)
    audio = recording.astype(np.float32).flatten()
    return pipe(audio)["text"]

def run_chatbot(user_input):
    return agent.run(user_input)

def main():
    global q, stop_recording
    print("Welcome to the chatbot! Say 'exit' to end the conversation.")
    speak("Welcome to the chatbot! You can start speaking now.")

    while True:
        q = queue.Queue()
        stop_recording = threading.Event()

        threading.Thread(target=record_audio, daemon=True).start()
        input("Press Enter to stop recording...")
        stop_recording.set()

        print("Transcribing...")
        user_input = transcribe_audio()
        print(f"You said: {user_input}")

        if user_input.lower() == 'exit':
            print("Goodbye!")
            speak("Goodbye!")
            break

        if "ลาก่อน" in user_input:
            print("Goodbye!")
            speak("Goodbye!")
            break

        try:
            response = run_chatbot(user_input)
            print("Chatbot:", response)
            speak(response)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            print(error_message)
            speak(error_message)

    pygame.quit()

if __name__ == "__main__":
    main()
