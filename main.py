#import os
import asyncio
from threading import Thread
import time
from queue import Queue
import re
import logging

from fastapi import HTTPException, BackgroundTasks, FastAPI, UploadFile, File, Body
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from pydantic import BaseModel, ConfigDict, HttpUrl, FileUrl, FilePath, model_validator
from typing import Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
import requests
from starlette.responses import RedirectResponse

from helpers import *
import torch
import torchaudio
from deepmultilingualpunctuation import PunctuationModel
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
import faster_whisper
from nemo.collections.asr.models.msdd_models import NeuralDiarizer

from utils import process_list, upload_files_concurrently


@dataclass
class Models:
    whisper_pipeline: Any
    msdd_model: Any
    alignment_model: Any
    alignment_tokenizer: Any
    alignment_dictionary: Any
    punct_model: Any

@dataclass
class Configs:
    TEMP_PATH: str
    BATCH_SIZE: int
    DEVICE: str
    WHISPER_MODEL: str
    TIME_TO_REMOVE: int

@dataclass
class ReturnMessage:
    message:str

class TranscriptionResponse(BaseModel):
    text: str
    filename: str
    metadata: dict
    token: str

class TranscriptionStatus(BaseModel):
    creation_time: datetime
    status:str
    result:list[str]

class Webhook(BaseModel):
    url: HttpUrl
    metadata: Optional[dict] = {}
    token: Optional[str] = ""
    model_config = ConfigDict(extra='forbid')

class TranscribeParams(BaseModel):
    model_config = ConfigDict(extra='forbid')
    language: Optional[str] = "es"
    stemming: Optional[bool] = False
    webhook: Optional[Webhook] = None
    @model_validator(mode='before')
    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value

class TranscribeParamsUri(TranscribeParams):
    audio_uri: list[FilePath | FileUrl]

logging.getLogger().setLevel(logging.INFO)
tags_metadata = [
    {
        "name": "processing",
        "description": "Audio files transcription process",
    },
    {
        "name": "status",
        "description": "Audio files transcription status",
    },
]

description = """
Diarization service that transcribe audio files, specifying speakers. 🚀
Return a .srt and .txt files.
"""

AUDIO_PATH = '/home/administrador/audio2'
STATIC_PATH = os.path.join(AUDIO_PATH, "static")
os.makedirs(STATIC_PATH, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    Thread(target=transcription_worker, daemon=True).start()
    yield  #Execution after closing app
    print("exit")


app = FastAPI(
    title="Whisper diarization service",
    description=description,
    summary="Transcribe audio files to srt format",
    version="1.0.0",
    contact={
        "name": "Javier Fernández",
        #"url": "https://github.com/MahmoudAshraf97/whisper-diarization",
        "email": "jfernandez@iter.es",
    },
    license_info={
        "name": "Apache 2.0",
        "identifier": "MIT",
    }, openapi_tags=tags_metadata,
    lifespan=lifespan
)
STATIC_ROUTE = "/transcriptions"
app.mount(STATIC_ROUTE, StaticFiles(directory=STATIC_PATH), name="static")

trancription_tasks = {}
trancription_tasks_queue = Queue()

configs = Configs(
    TEMP_PATH=os.path.join(AUDIO_PATH, "temp_output"),
    BATCH_SIZE=32,
    DEVICE="cuda" if torch.cuda.is_available() else "cpu",
    WHISPER_MODEL="mobiuslabsgmbh/faster-whisper-large-v3-turbo",  #large-v3",
    TIME_TO_REMOVE=259200 #3 days in seconds: 60 x 60 x 24 x 3
)

models = Models(
    whisper_pipeline=None,
    msdd_model=None,
    alignment_model=None,
    alignment_tokenizer=None,
    alignment_dictionary=None,
    punct_model=None
)


def load_models() -> None:
    #print("loading")
    global models
    global configs

    whisper_model = faster_whisper.WhisperModel(
        configs.WHISPER_MODEL, device=configs.DEVICE, compute_type="float16" if configs.DEVICE == "cuda" else "int8"
    )
    models.whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
    models.alignment_model, models.alignment_tokenizer = load_alignment_model(
        configs.DEVICE,
        dtype=torch.float16 if configs.DEVICE == "cuda" else torch.float32,
    )
    models.msdd_model = NeuralDiarizer(cfg=create_config(configs.TEMP_PATH)).to(configs.DEVICE)
    models.punct_model = PunctuationModel(model="kredor/punctuate-all")


def stemming(audio_file: str) -> str:
    """
    Isolate vocals from the rest of the audio
    """
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


def nemo_process(audio_waveform, temp_path):
    global models

    create_config(temp_path)
    #sound = AudioSegment.from_file(audio_file).set_channels(1)
    #sound.export(os.path.join(temp_path, "mono_file.wav"), format="wav")
    #models.msdd_model.diarize()  #diarize all in temp_path
    torchaudio.save(
        os.path.join(temp_path, "mono_file.wav"),
        torch.from_numpy(audio_waveform).unsqueeze(0).float(),
        16000,
        channels_first=True,
    )
    models.msdd_model.diarize()  # diarize all in temp_path


def diarize(audio_file: str, lang: str, is_stemming: bool) -> str:
    global configs
    global models
    inicio = time.time()
    os.makedirs(configs.TEMP_PATH, exist_ok=True)
    vocal_target = audio_file
    if is_stemming:
        vocal_target = stemming(vocal_target)
    #return "Hola mundo"
    language = process_language_arg(lang, configs.WHISPER_MODEL)

    audio_waveform = faster_whisper.decode_audio(vocal_target)
    proc = Thread(target=nemo_process, args=(audio_waveform, configs.TEMP_PATH))
    proc.start()

    transcript_segments, info = models.whisper_pipeline.transcribe(
        audio_waveform,
        language,
        suppress_tokens=([-1]),  #no supress numeral
        batch_size=configs.BATCH_SIZE,
        # without_timestamps=True,
    )

    full_transcript = "".join(segment.text for segment in transcript_segments)
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    tokens_starred, text_starred = preprocess_text(full_transcript, romanize=True,
                                                   language=langs_to_iso[info.language], )

    emissions, stride = generate_emissions(
        models.alignment_model,
        torch.from_numpy(audio_waveform).to(models.alignment_model.dtype).to(models.alignment_model.device),
        batch_size=configs.BATCH_SIZE,
    )
    # if torch.cuda.is_available():
    #     torch.cuda.empty_cache()
    segments, scores, blank_token = get_alignments(emissions, tokens_starred, models.alignment_tokenizer, )

    spans = get_spans(tokens_starred, segments, blank_token)
    word_timestamps = postprocess_results(text_starred, spans, stride, scores)
    #print("waiting...")
    proc.join()  #wait for nemo process

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    speaker_ts = []
    with open(os.path.join(configs.TEMP_PATH, "pred_rttms", "mono_file.rttm"), "r") as f:
        lines = f.readlines()
        for line in lines:
            line_list = line.split(" ")
            s = int(float(line_list[5]) * 1000)
            e = s + int(float(line_list[8]) * 1000)
            speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])

    wsm = get_words_speaker_mapping(word_timestamps, speaker_ts, "start")

    if info.language in punct_model_langs:
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        logging.warning(
            f"Punctuation restoration is not available for {info.language} language. Using the original punctuation."
        )

    wsm = get_realigned_ws_mapping_with_punctuation(wsm)
    ssm = get_sentences_speaker_mapping(wsm, speaker_ts)

    fin = time.time()
    tiempo_ejecucion = fin - inicio
    logging.info(f"Transcription time of {audio_file}: {tiempo_ejecucion} seconds")

    output_path = os.path.join(STATIC_PATH, os.path.basename(vocal_target))
    with open(f"{os.path.splitext(output_path)[0]}.txt", "w", encoding="utf-8-sig") as f:
        get_speaker_aware_transcript(ssm, f)

    with open(f"{os.path.splitext(output_path)[0]}.srt", "w", encoding="utf-8-sig") as srt:
        write_srt(ssm, srt)

    cleanup(configs.TEMP_PATH)  #comment for get embeddings
    cleanup(audio_file)
    return getPlainSRT(ssm)


def send_results(plain_text: str, file_path: str, webhook:Webhook) -> bool:
    global configs
    #print(url, token, metadata)
    filename = os.path.basename(file_path)
    try:
        payload = TranscriptionResponse(text=plain_text.replace("\"", "\'"), filename=filename,token=webhook.token,metadata=webhook.metadata)
        req = requests.post(webhook.url, json=payload.model_dump())
        req.raise_for_status()
    except requests.exceptions.RequestException as e:
        #print("ERROR! " + str(e))
        logging.warning(f"Error sending {filename}: {str(e)}")
        return False

    return True

def transcription_worker() -> None:
    while True:
        current_params: dict = trancription_tasks_queue.get()
        audio_file = current_params["file_path"]
        filename = os.path.basename(current_params["file_path"])
        try:
            current_status: TranscriptionStatus = trancription_tasks[filename]
            current_status.status = "Transcribing..."
            result = diarize(audio_file, current_params.get("language", "es"), current_params.get("stemming", False))
            if result is not None and len(result) > 0:
                current_status.status = "Completed"
                file_result = f"{STATIC_ROUTE}/{os.path.splitext(filename)[0]}"
                current_status.result= [file_result +".srt",file_result+".txt"]

                webhook:Webhook = current_params.get("webhook")
                if webhook is not None:
                    srt_filename= os.path.splitext(filename)[0] + ".srt"
                    was_sent = send_results(result, srt_filename, webhook)
                    if was_sent:
                        current_status.status = "Completed | Sending successfully"
                    else:
                        current_status.status = "Completed | Sending fails"

            else:
                current_status.status = "Failed"

        except Exception as e:
            #print("Failed:", str(e))
            logging.error("Failed: " + str(e))
            current_status:TranscriptionStatus = trancription_tasks[filename]
            current_status.status="Failed:" + str(e)

        finally:
            trancription_tasks_queue.task_done()
            #os.remove(tmp_path)


async def cleanup_task(task_id: str) -> None:
    """
    Remove task in queue and files in static directory
    """
    global  configs
    await asyncio.sleep(configs.TIME_TO_REMOVE)
    trancription_tasks.pop(task_id, None)
    basename_file = os.path.splitext(task_id)[0]
    output_path = os.path.join(STATIC_PATH, basename_file)
    cleanup(output_path + ".srt")
    cleanup(output_path + ".txt")
    logging.info("Files removed: " +output_path +".srt - " + output_path+".txt")

@app.get("/", include_in_schema=False)
async def main():
    return RedirectResponse(url='/docs')

@app.get("/status/", tags=["status"])
async def status() -> dict:
    """
    Get status of all audio files in process.
    """
    return trancription_tasks


@app.get("/status/{audio_file}", tags=["status"])
async def get_task_status(audio_file: str) -> TranscriptionStatus:
    """
    Get status of an audio file in process.
    - audio_file: filename with extension of an audio file
    """
    task:TranscriptionStatus = trancription_tasks.get(audio_file)
    if not task:
        raise HTTPException(status_code=404, detail="Audio file not found!")
    return task

@app.post("/transcribe_uri/", tags=["processing"])
async def transcribe_uri(params: TranscribeParamsUri, background_tasks: BackgroundTasks) -> ReturnMessage:
    """
    Transcribe **.mp3 or .wav** audio format file to srt file. Parameters description:

    - audio_path : List of existing local full path audios or reachable url audio files.
    - language: Current audio language.(Optional, default "es").
    - stemming: True if is a audio music, false otherwise. It separates music signal of vocal. (Optional, default False).
    - webhook: Send transcription response (srt) to a selected endpoint (Optional, default None):
        - url: Valid reachable url for sending response.
        - metadata: Json with custom metadata. (Optional, default empty json).
        - token: Required token for sending to endpoint  (Optional, default empty string).
    """
    #TODO validate audio format (mp3, wav,)
    valid_files = process_list(params.audio_uri)
    errors_count = len(params.audio_uri) - len(valid_files)
    valid_files_names = []
    for file_path in valid_files:
        #print("Queueing a job")
        filename = os.path.basename(file_path)
        valid_files_names.append(filename)
        trancription_tasks[filename] = TranscriptionStatus(creation_time=datetime.now(), status="Enqueued", result=[])
        #validate configs
        trancription_tasks_queue.put({"file_path": file_path, "webhook": params.webhook,
                                      "language": params.language, "stemming": params.stemming})

        background_tasks.add_task(cleanup_task, filename)

    valid_file_names = "".join(valid_files_names)
    return ReturnMessage(message=f"Processing {str(len(valid_files))} files: {valid_file_names}, Files wrong: {str(errors_count)}")


@app.post("/transcribe/", tags=["processing"])
async def transcribe_audio_file(background_tasks: BackgroundTasks,  files:List[UploadFile]=File(description="Files to transcribe"),
                          params:TranscribeParams=Body(...)) -> ReturnMessage:
    """
    Transcribe audio files in **.mp3** or **.wav** into a .srt format.
    `TODO: Add support for other audio formats.`

    - files: Audio files in mp3 format
    - language: Current audio language.(Optional, default "es").
    - stemming: True if is a audio music, false otherwise. It separates music signal of vocal. (Optional, default False).
    - webhook: Send transcription response (srt) to a selected endpoint (Optional, default None):
        - url: Valid reachable url for sending response.
        - metadata: Json with custom metadata {"name": ["some", "more"], "category": ["only_one"]}. (Optional, default empty json).
        - token: Required token for sending to endpoint  (Optional, default empty string).
    """
    # https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request
    valid_files = upload_files_concurrently(files)
    valid_files_names = []
    for file_path in valid_files:
        #print("Queueing a job")
        filename = os.path.basename(file_path)
        valid_files_names.append(filename)
        #validate configs
        trancription_tasks[filename] = TranscriptionStatus(creation_time=datetime.now(), status="Enqueued", result=[])

        trancription_tasks_queue.put({"file_path": file_path, "webhook": params.webhook,
                                      "language": params.language, "stemming": params.stemming})

        background_tasks.add_task(cleanup_task, filename)

    valid_file_names = "".join(valid_files_names)
    return ReturnMessage(message=f"Processing {str(len(valid_files))} files: {valid_file_names}, Files wrong: {str(len(files) - len(valid_files))}")

@app.webhooks.post("transcription_response")
def new_transcription(body: TranscriptionResponse):
    """
    It will send the srt transcription in plain text of the audio file  in a POST request to the URL that user provided in the transcribe webhook parameter settings.
    """
    pass
