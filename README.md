##Whisper diarization FASTAPI 
based on project https://github.com/MahmoudAshraf97/whisper-diarization

Queue audio an process sequentially

# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg
sudo apt update && sudo apt install cython3
pip install -r requirements.txt

#Docker 
build: sudo docker build -t  hub.iter.es/diarization_api:V0.4 .
dev: sudo docker run  -it --gpus all --network host -v /home/administrador/audio2:/home/administrador/audio2 hub.iter.es/diarization_api:V0.4 
pro: sudo docker run  -d --gpus all --network host -v /home/administrador/audio2:/home/administrador/audio2 hub.iter.es/diarization_api:V0.4 

