## 🎙️ Whisper Audio Splitter & Transcriber
Este script optimiza la transcripción de audios largos (3h+) utilizando Faster Whisper y aceleración por hardware NVIDIA CUDA.

---

## 🚀 Optimización para RTX Serie 40
- **División Automática**: Fragmenta audios pesados para evitar desbordamiento de memoria VRAM.

- **Gestión de Memoria**: Implementa gc.collect() y torch.cuda.empty_cache() para liberar la GPU después de cada proceso.

- **Precisión**: Configurado con float16 para máximo rendimiento en núcleos Tensor.

---

## 🛠️ Requisitos
*Tener instalada la suite de FFmpeg (necesaria para pydub).*

*Drivers de CUDA configurados.*

---
- Nota: En la primera ejecución, el script descargará el modelo de Hugging Face. Se recomienda activar el Modo Desarrollador en Windows para una gestión de caché óptima.