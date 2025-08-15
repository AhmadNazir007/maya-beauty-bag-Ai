# twilio_voice.py

from fastapi import APIRouter, FastAPI, Form, File, UploadFile
from fastapi.responses import PlainTextResponse, JSONResponse
from twilio.twiml.voice_response import VoiceResponse
from backend.openai_logic import generate_maya_reply

import whisper
import os


app = FastAPI()  # ✅ This is the object uvicorn needs

router = APIRouter()

@app.post("/voice", response_class=PlainTextResponse)
async def handle_call(SpeechResult: str = Form(default="")):
    vr = VoiceResponse()

    if not SpeechResult:
        # First prompt
        vr.say("Hey love, I'm Maya. This is your moment. What beauty vibe are you going for today?", voice='Polly.Joanna')
        vr.gather(input='speech', timeout=5)
    else:
        # Generate reply from OpenAI
        maya_reply = generate_maya_reply(SpeechResult)
        vr.say(maya_reply, voice='Polly.Joanna')
        vr.gather(input='speech', timeout=5)

    return str(vr)

model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    audio_path = f"temp_{audio.filename}"
    with open(audio_path, "wb") as f:
        f.write(await audio.read())

    result = model.transcribe(audio_path)
    os.remove(audio_path)

    return JSONResponse({"text": result["text"]})

