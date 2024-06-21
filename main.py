import os
import asyncio
from threading import Thread
from queue import Queue
import re
import logging

from fastapi import HTTPException, BackgroundTasks, FastAPI, status
from pydantic import BaseModel, ConfigDict
from typing import List, Any, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import requests

from helpers import *
import torch
from pydub import AudioSegment
from deepmultilingualpunctuation import PunctuationModel
from ctc_forced_aligner import (
    load_alignment_model,
    generate_emissions,
    preprocess_text,
    get_alignments,
    get_spans,
    postprocess_results,
)
import whisperx
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

@dataclass
class Models:
    whisper_model: Any
    msdd_model: Any
    alignment_model: Any
    alignment_tokenizer: Any
    alignment_dictionary: Any
    punct_model: Any

@dataclass
class Configs:
    TEMP_PATH:str
    BATCH_SIZE:int
    DEVICE:str
    WHISPER_MODEL:str
    LANGUAGE:str
    DATA_SERVER_URL:str
    STEMMING:bool
 
class TranscribeParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    configs: Optional[Dict] = None
    audio_path: List[str]
    
tags_metadata = [
    {
        "name": "processing",
        "description": "Process audio files",
    },
    {
        "name": "status",
        "description": "Show status",
    },
]

description = """
Transcribe a audio file into text, specifing speakers. 🚀
Return a srt file.
* OpenAI Whisper for transcribing
* Nvidia Nemo (ASR) for splitting and clustering 
"""

app = FastAPI(
    title="Whisper diarization service",
    description=description,
    summary="Transcribe audio files to srt format",
    version="1.0.0",
    contact={
        "name": "Javier Fernández",
        "url": "https://github.com/MahmoudAshraf97/whisper-diarization",
        "email": "jfernandez@iter.es",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    },openapi_tags=tags_metadata)

trancription_tasks = {}
trancription_tasks_queue = Queue()

ROOT = '/home/administrador/audio2'
configs = Configs(    
    TEMP_PATH = os.path.join(ROOT, "temp_output"),
    BATCH_SIZE = 32,
    DEVICE = "cuda", #or "cpu"
    WHISPER_MODEL = "large-v3",
    LANGUAGE = None, #None for autodetection
    DATA_SERVER_URL = "http://0.0.0.0:8000/documents/upload/plainSRT",
    STEMMING = False
)

models = Models(
    whisper_model=None,
    msdd_model=None,
    alignment_model=None,
    alignment_tokenizer=None,
    alignment_dictionary=None,
    punct_model=None
)

def load_models():
    #print("loading")
    global models
    global configs
    
    models.whisper_model = whisperx.load_model(
        configs.WHISPER_MODEL,
        configs.DEVICE,
        compute_type="float16" if configs.DEVICE == "cuda" else "int8",
        asr_options={"suppress_numerals": False},
    )
    models.alignment_model, models.alignment_tokenizer, models.alignment_dictionary = load_alignment_model(
        configs.DEVICE,
        dtype=torch.float16 if configs.DEVICE == "cuda" else torch.float32,
    )
    models.msdd_model = NeuralDiarizer(cfg=create_config(configs.TEMP_PATH)).to(configs.DEVICE)
    models.punct_model = PunctuationModel(model="kredor/punctuate-all")
    #print("end loading")

# Isolate vocals from the rest of the audio
def stemming(audio_file:str):
    vocal_tarjet = audio_file
    global configs

    return_code = os.system(
        f'python3 -m demucs.separate -n htdemucs --two-stems=vocals "{vocal_tarjet}" -o "{configs.TEMP_PATH}"'
    )

    if return_code != 0:
        logging.warning(
            "Source splitting failed, using original audio file. Use --no-stem argument to disable it."
        )
        vocal_target = audio_file
    else:
        vocal_target = os.path.join(
            configs.TEMP_PATH,
            "htdemucs",
            os.path.splitext(os.path.basename(audio_file))[0],
            "vocals.wav",
        )
    
    return vocal_target

def transcribe_batched(audio_file:str):
    global models
    
    audio = whisperx.load_audio(audio_file)
    result = models.whisper_model.transcribe(audio, language=configs.LANGUAGE, batch_size=configs.BATCH_SIZE)
    
    return result["segments"], result["language"], audio

def nemo_process(audio_file, temp_path):
    global models
    
    create_config(temp_path)
    sound = AudioSegment.from_file(audio_file).set_channels(1)
    sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
    models.msdd_model.diarize()    #diarize all in temp_path

def diarize(audio_file):
    global configs
    global models

    os.makedirs(configs.TEMP_PATH, exist_ok=True)
    vocal_target = audio_file
    if configs.STEMMING:
        vocal_target = stemming(vocal_target)
    proc = Thread(target=nemo_process, args=(vocal_target, configs.TEMP_PATH))
    proc.start()
    
    whisper_results, language, audio_waveform = transcribe_batched(vocal_target)
    audio_waveform = (torch.from_numpy(audio_waveform).to(models.alignment_model.dtype).to(models.alignment_model.device))
    emissions, stride = generate_emissions(models.alignment_model, audio_waveform, batch_size=configs.BATCH_SIZE)
    full_transcript = "".join(segment["text"] for segment in whisper_results)
    #print(full_transcript)
    tokens_starred, text_starred = preprocess_text(full_transcript, romanize=True, language=langs_to_iso[language],)
    
    segments, scores, blank_id = get_alignments(emissions, tokens_starred, models.alignment_dictionary,)
    spans = get_spans(tokens_starred, segments, models.alignment_tokenizer.decode(blank_id))
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    #print("waiting...")
    proc.join()  #wait for nemo process

    speaker_ts = []
    with open(os.path.join(configs.TEMP_PATH, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")
    
    if language in punct_model_langs:
    # restoring punctuation in the transcript to help realign the sentences

        words_list = list(map(lambda x: x["word"], wsm))

        labled_words = models.punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"

        # We don't want to punctuate U.S.A. with a period. Right?
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labled_words):
            word = word_dict["word"]
            if (
                word
                and labeled_tuple[1] in ending_puncts
                and (word[-1] not in model_puncts or is_acronym(word))
            ):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

    else:
        logging.warning(
            f"Punctuation restoration is not available for {language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    # with open(f"{os.path.splitext(vocal_target)[0]}.txt", "w", encoding="utf-8-sig") as f:    
        # get_speaker_aware_transcript(ssm, f)

    # with open(f"{os.path.splitext(vocal_target)[0]}1.srt", "w", encoding="utf-8-sig") as srt:
       # write_srt(ssm, srt)

    cleanup(configs.TEMP_PATH)
    cleanup(audio_file)
    return getPlainSRT(ssm)

def send_results(plain_text:str, metadata:dict, file_path:str):
    global configs

    try:
        req = requests.post(configs.DATA_SERVER_URL, json={"text": plain_text.replace("\"", "\'"), "metadata": metadata, "filename":os.path.basename(file_path)})  #encode texts semantic search api
        req.raise_for_status()
    except requests.exceptions.RequestException as e:
        print("ERROR! "+ str(e))
        with open(f"{os.path.splitext(file_path)[0]}.srt", "w", encoding="utf-8-sig") as srt:
            srt.write(plain_text)

def transcription_worker() -> None:
    while True:
        audio_file = trancription_tasks_queue.get()
        filename = os.path.basename(audio_file)
        try:
            result = diarize(audio_file)
            send_results(result, {}, audio_file)
            trancription_tasks[filename].update({"status": "completed", "result": result})

        except Exception as e:
            trancription_tasks[filename].update({"status": "failed", "result": str(e)})

        finally:
            trancription_tasks_queue.task_done()
            #os.remove(tmp_path)

async def cleanup_task(task_id: str) -> None:
    await asyncio.sleep(60 * 60)
    trancription_tasks.pop(task_id, None)

@app.on_event("startup")
async def start_queue():
    load_models()
    Thread(target=transcription_worker, daemon=True).start()

@app.get("/status",tags=["status"])
async def status() -> dict:
    return trancription_tasks

@app.get("/status/{audio_file}",tags=["status"])
async def get_task_status(audio_file: str) -> dict:
    task = trancription_tasks.get(audio_file)
    if not task:
        raise HTTPException(status_code=404, detail="Audio file not found!")

    return {
        "audio_file": audio_file,
        "creation_time": task["creation_time"],
        "status": task["status"]
    }

@app.post("/transcribe", tags=["processing"])
async def transcribe(params:TranscribeParams, background_tasks: BackgroundTasks):
    '''
    Transcribe audio file to srt file. Admit an absolute path where the audio file is located.
    '''
    count = 0
    for file_path in params.audio_path:
        if os.path.isfile(file_path):
            #print("Queueing a job")
            filename = os.path.basename(file_path)
            trancription_tasks[filename] = {
                "status": "loading",
                "creation_time": datetime.now(),
                "result": None
            }
            #validate configs
            trancription_tasks[filename].update({"status": "processing"})
            trancription_tasks_queue.put(file_path)

            background_tasks.add_task(cleanup_task, filename)
        else:
            count += 1

    return {"result": "Files processing: " + str(len(params.audio_path) - count)+ ", errors: " +str(count)}

