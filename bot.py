import os
import asyncio
import ffmpeg
from aiogram import Bot, Dispatcher, Router, types, F
from aiogram.filters import Command
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from transformers import pipeline
import torch

# ←←← ТВОЙ ТОКЕН (уже вставлен)
BOT_TOKEN = "8591454671:AAFbv7OcySJWvNxEEj3XjdV01TZHfbj9nZY"

# ←←← Путь к твоей дообученной казахской модели
MODEL_PATH = r"C:\Users\User\.vscode\wawtowec_kazakh_gpu"

print("Загружаю твою казахскую Whisper-модель... (может занять 20–60 сек)")
asr = pipeline(
    "automatic-speech-recognition",
    model=MODEL_PATH,
    device=0 if torch.cuda.is_available() else -1,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

# Правильный способ задать parse_mode в новых версиях aiogram
bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
router = Router()
dp.include_router(router)


def ogg_to_wav(ogg_path: str) -> str:
    wav_path = ogg_path.replace(".ogg", ".wav")
    ffmpeg.input(ogg_path).output(
        wav_path, ar=16000, ac=1, loglevel="quiet"
    ).overwrite_output().run()
    return wav_path


@router.message(Command("start"))
async def start(message: types.Message):
    await message.answer(
        "Привет! Отправь голосовое сообщение или аудиофайл — я мгновенно переведу казахскую речь в текст (твоя дообученная модель + GPU)"
    )


# Исправленный фильтр: ловим сообщения с VOICE, AUDIO или DOCUMENT с помощью F
@router.message(F.voice | F.audio | F.document)
async def handle_audio(message: types.Message):
    await message.answer("Скачиваю и распознаю...")

    # Определяем файл
    if message.voice:
        file_id = message.voice.file_id
    elif message.audio:
        file_id = message.audio.file_id
    elif message.document and message.document.mime_type and "audio" in message.document.mime_type:
        file_id = message.document.file_id
    else:
        await message.reply("Пожалуйста, отправьте голосовое сообщение или аудиофайл")
        return

    file = await bot.get_file(file_id)
    ogg_path = f"temp_{message.message_id}.ogg"
    await bot.download_file(file.file_path, ogg_path)
    wav_path = ogg_to_wav(ogg_path)

    try:
        result = asr(wav_path, chunk_length_s=30, generate_kwargs={"language": "kazakh", "task": "transcribe"})
        text = result["text"].strip()
        if not text:
            text = "Тишина или ничего не распознал"
        await message.reply(text)
    except Exception as e:
        await message.reply(f"Ошибка распознавания: {e}")
    finally:
        for p in [ogg_path, wav_path]:
            try:
                os.remove(p)
            except:
                pass


async def main():
    device = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    print(f"Бот запущен и работает на {device}!")
    print("Готов принимать казахскую речь")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())