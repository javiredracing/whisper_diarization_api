import os
import concurrent.futures
import requests
from urllib.parse import urlparse

AUDIO_PATH = "/home/administrador/audio2"

def process_item(item):
    if is_url(item):
        return download_file(item)
    elif os.path.isfile(item):
        return item
    else:
        return None


def is_url(string):
    try:
        result = urlparse(string)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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


def process_list(input_list):
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_item, input_list))

    return [result for result in results if result is not None]