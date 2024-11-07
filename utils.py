import os
import concurrent.futures
import shutil
from concurrent.futures import as_completed

import requests
from urllib.parse import urlparse
import validators

AUDIO_PATH = '/home/administrador/audio2'


def process_item(item):
    if validators.url(item):
        return download_file(item)
    elif os.path.isfile(item):
        return item
    else:
        return None


#def is_url(string):
    # try:
    #     #result = urlparse(string)
    #     return all([result.scheme, result.netloc])
    # except AttributeError:
    #     return False



def download_file(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            filename = os.path.join(AUDIO_PATH, os.path.basename(url))
            with open(filename, 'wb') as file:
                file.write(response.content)
            return filename
    except requests.RequestException:
        pass
    return None

def write_file(file):
    """Función para procesar un único archivo."""
    if file.content_type.startswith("audio/"):
        audio_file = os.path.join(AUDIO_PATH, file.filename)
        if not os.path.isfile(audio_file):  # Si no existe
            try:
                with open(audio_file, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                    return audio_file  # Retorna el archivo válido
            except Exception as e:
                print(f"There was an error uploading the file: {e}")
            finally:
                file.file.close()
    return None  # Retorna None si no es un archivo válido


def upload_files_concurrently(files):
    """Sube archivos de forma concurrente."""
    valid_files = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Enviar las tareas al pool de hilos
        futures = {executor.submit(write_file, file): file for file in files}

        # Obtener los resultados a medida que se completan
        for future in as_completed(futures):
            result = future.result()  # Obtener el resultado de la tarea
            if result:  # Si el resultado no es None, es un archivo válido
                valid_files.append(result)

    return valid_files

def process_list(input_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, input_list))

    return [result for result in results if result is not None]