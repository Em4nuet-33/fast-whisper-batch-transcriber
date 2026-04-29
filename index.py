from faster_whisper import WhisperModel
import torch
import gc
from pydub import AudioSegment
import math
import os

# --- CONFIGURACIÓN ---
model_size = "small"   # tamaño del modelo
device = "cuda"
compute_type = "float16"  # Ideal para RTX serie 40

# --- CARGAR MODELO ---
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# --- DIVIDIR AUDIO EN FRAGMENTOS ---
def dividir_audio(archivo, duracion_min=30):
    audio = AudioSegment.from_file(archivo)
    duracion_ms = len(audio)
    fragmento_ms = duracion_min * 60 * 1000
    num_fragmentos = math.ceil(duracion_ms / fragmento_ms)

    archivos = []
    for i in range(num_fragmentos):
        inicio = i * fragmento_ms
        fin = min((i+1) * fragmento_ms, duracion_ms)
        fragmento = audio[inicio:fin]
        nombre = f"fragmento_{i+1}.mp3"
        fragmento.export(nombre, format="mp3")
        archivos.append(nombre)
    return archivos

# --- TRANSCRIPCIÓN POR FRAGMENTOS ---
def transcribir_fragmentos(archivos, salida="transcripcion.txt"):
    with open(salida, "w", encoding="utf-8") as f:
        for archivo in archivos:
            print(f"Transcribiendo {archivo}...")
            segments, info = model.transcribe(archivo, beam_size=5, language="es")
            for segment in segments:
                f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            # Liberar memoria entre fragmentos
            torch.cuda.empty_cache()
            gc.collect()
            # Borrar fragmento temporal
            os.remove(archivo)

# --- EJECUCIÓN ---
archivo_objetivo = "./samples/ejemplo.mp3"

if os.path.exists(archivo_objetivo):
    archivos = dividir_audio(archivo_objetivo, duracion_min=30)
    transcribir_fragmentos(archivos)
else:
    print(f" El archivo {archivo_objetivo} no se encontró. Coloca un mp3 para iniciar.")

# --- LIBERAR RECURSOS ---
del model
gc.collect()
torch.cuda.empty_cache()
print("Memoria de la RTX 4060 liberada correctamente. Fragmentos temporales eliminados.")
