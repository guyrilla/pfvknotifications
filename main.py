import os
import sys
import time
import random
import threading
import numpy as np
import soundfile as sf
import soundcard as sc
import requests
from scipy.signal import fftconvolve, resample_poly
from math import gcd
from dotenv import load_dotenv

load_dotenv()

VK_TOKEN = os.environ.get("VK_COMMUNITY_TOKEN")
VK_PEER_ID = os.environ.get("VK_PEER_ID")
VK_API_VER = os.environ.get("VK_API_VERSION", "5.199")
VK_GROUP_ID = os.environ.get("VK_GROUP_ID", "")
# Интервал между сообщениями в спам-режиме (сек). 0.5 = очень агрессивно.
SPAM_INTERVAL = float(os.environ.get("VK_SPAM_INTERVAL", "0.5"))

if not VK_TOKEN:
    sys.exit("Ошибка: VK_COMMUNITY_TOKEN не задан в .env")
if not VK_PEER_ID:
    sys.exit("Ошибка: VK_PEER_ID не задан в .env")
if len(sys.argv) < 2:
    sys.exit('Использование: python detector.py "текст сообщения для VK"')

MESSAGE = sys.argv[1]

# ==== Детектор ====
TARGET_FILE = "sound.wav"
THRESHOLD = 0.6
COOLDOWN = 2.0
BLOCK = 4096
TARGET_SR = 48000
# ==================================


def send_vk(text: str, return_msg_id: bool = False):
    """
    Отправляет сообщение. Возвращает msg_id (или True) при успехе, None при ошибке.
    При flood control (ошибка 6) — ждёт и повторяет до 5 раз.
    """
    url = "https://api.vk.com/method/messages.send"
    base_payload = {
        "access_token": VK_TOKEN,
        "v": VK_API_VER,
        "peer_id": VK_PEER_ID,
        "message": text,
        "group_id": VK_GROUP_ID,
        "disable_mentions": 1,
    }
    base_payload = {k: v for k, v in base_payload.items() if v != ""}

    for retry in range(5):
        payload = dict(base_payload)
        payload["random_id"] = random.randint(0, 2**31 - 1)
        try:
            r = requests.post(url, data=payload, timeout=10)
            data = r.json()
            if "response" in data:
                msg_id = data["response"]
                return msg_id if return_msg_id else True

            err = data.get("error", {})
            code = err.get("error_code")
            msg = err.get("error_msg")

            if code == 6:
                print(f"[!] Flood control, пауза 5 сек (попытка {retry + 1}/5)", flush=True)
                time.sleep(5)
                continue

            hints = {
                5: "неверный/просроченный токен",
                901: "пользователь не писал сообществу",
                902: "пользователь запретил сообщения от сообщества",
                945: "беседа закрыта",
                946: "сообщество не разрешает сообщения",
                910: "бот не может писать первым",
                100: "неверный peer_id / параметры",
            }
            hint = hints.get(code, "")
            print(
                f"[!] VK ошибка {code}: {msg}  {('— ' + hint) if hint else ''}",
                flush=True,
            )
            return None
        except Exception as e:
            print(f"[!] Ошибка отправки в VK: {e}", flush=True)
            return None
    return None


def check_status(peer_id, first_msg_id, latest_msg_id):
    """
    Одна проверка через getHistory.
    Возвращает (user_replied, our_read) либо None при ошибке сети.
    """
    url = "https://api.vk.com/method/messages.getHistory"
    payload = {
        "access_token": VK_TOKEN,
        "v": VK_API_VER,
        "peer_id": peer_id,
        "count": 50,
    }
    try:
        r = requests.post(url, data=payload, timeout=10)
        data = r.json()
        if "response" not in data:
            err = data.get("error", {})
            print(
                f"[!] getHistory {err.get('error_code')}: {err.get('error_msg')}",
                flush=True,
            )
            return None
        items = data["response"].get("items", [])
        user_replied = False
        our_read = False
        for m in items:
            mid = m.get("id", 0)
            # out=0 → входящее сообщение от пользователя
            if m.get("out") == 0 and mid > first_msg_id:
                user_replied = True
            if mid in (first_msg_id, latest_msg_id) and m.get("read_state") == 1:
                our_read = True
        return user_replied, our_read
    except Exception as e:
        print(f"[!] getHistory исключение: {e}", flush=True)
        return None


def resend_until_read(first_msg_id, text):
    """
    Спам-цикл: пока сообщение не прочитано и пользователь не ответил —
    отправляем снова каждые SPAM_INTERVAL секунд.
    """
    latest_msg_id = first_msg_id
    attempt = 0
    consecutive_fails = 0

    while True:
        status = check_status(VK_PEER_ID, first_msg_id, latest_msg_id)

        if status is None:
            # API не ответил — не спамим зря, ждём и повторяем проверку
            time.sleep(1)
            continue

        user_replied, our_read = status

        if our_read:
            print(
                f"[✓] Сообщение прочитано. Дозвон остановлен (попыток: {attempt}).",
                flush=True,
            )
            return
        if user_replied:
            print(
                f"[✓] Пользователь написал в ответ. Дозвон остановлен (попыток: {attempt}).",
                flush=True,
            )
            return

        attempt += 1
        new_id = send_vk(text, return_msg_id=True)
        if new_id is None:
            consecutive_fails += 1
            if consecutive_fails >= 10:
                print("[!] Слишком много ошибок отправки — дозвон остановлен.", flush=True)
                return
            time.sleep(3)
            continue
        consecutive_fails = 0
        latest_msg_id = new_id
        print(f"[📨] Спам #{attempt} (msg_id={new_id})", flush=True)
        time.sleep(SPAM_INTERVAL)


def send_async(text: str) -> None:
    """Отправляет первое сообщение и запускает дозвон в отдельном потоке."""
    def worker():
        first_msg_id = send_vk(text, return_msg_id=True)
        if first_msg_id is None:
            print("[!] Первая отправка не удалась — дозвон не запущен.", flush=True)
            return
        print(f"[+] Запущен дозвон до прочтения (msg_id={first_msg_id})", flush=True)
        resend_until_read(first_msg_id, text)

    threading.Thread(target=worker, daemon=True).start()


# ---- Эталон ----
target, orig_sr = sf.read(TARGET_FILE, dtype="float32")
if target.ndim > 1:
    target = target.mean(axis=1)
if orig_sr != TARGET_SR:
    g = gcd(orig_sr, TARGET_SR)
    target = resample_poly(target, TARGET_SR // g, orig_sr // g).astype(np.float32)
    print(f"[*] Ресемплинг эталона: {orig_sr} → {TARGET_SR} Hz")

sr = TARGET_SR
target_energy = float(np.sum(target.astype(np.float64) ** 2))
target_rev = target[::-1].astype(np.float64)


def find_loopback_mic(speaker):
    try:
        mic = sc.get_microphone(id=str(speaker.name), include_loopback=True)
        if mic is not None:
            return mic
    except Exception:
        pass
    try:
        for m in sc.all_microphones(include_loopback=True):
            if getattr(m, "isloopback", False):
                return m
    except Exception:
        pass
    try:
        for m in sc.all_microphones(include_loopback=True):
            n = str(m.name).lower()
            if ".monitor" in n or n.startswith("monitor of"):
                return m
    except Exception:
        pass
    raise RuntimeError("Loopback-микрофон не найден.")


speaker = sc.default_speaker()
mic = find_loopback_mic(speaker)
print(f"[*] Слушаю: {mic.name}")
print(f"[*] При детекте отправлю: «{MESSAGE}» → peer_id={VK_PEER_ID}")
print(f"[*] Спам-интервал: {SPAM_INTERVAL} сек (настраивается через VK_SPAM_INTERVAL)")
print(f"[*] Стоп: когда прочитано ИЛИ пользователь напишет любой ответ.")
print(f"[*] Готов. Ctrl+C для остановки.")

buf = np.zeros(len(target) + BLOCK * 2, dtype=np.float32)
last_trigger = 0.0

try:
    with mic.recorder(samplerate=sr, channels=1) as rec:
        while True:
            data = rec.record(numframes=BLOCK)
            if data.ndim > 1:
                data = data.mean(axis=1)
            data = data.astype(np.float32)
            if len(data) < BLOCK:
                data = np.pad(data, (0, BLOCK - len(data)))

            buf = np.concatenate([buf[len(data):], data])

            corr = fftconvolve(buf.astype(np.float64), target_rev, mode="valid")
            cum = np.concatenate([[0.0], np.cumsum(buf.astype(np.float64) ** 2)])
            lags = np.arange(len(corr))
            win_energy = cum[lags + len(target)] - cum[lags]
            win_energy = np.maximum(win_energy, 1e-9)
            norm_corr = corr / np.sqrt(win_energy * target_energy)

            peak = float(norm_corr.max())
            now = time.time()

            if peak >= THRESHOLD and (now - last_trigger) > COOLDOWN:
                last_trigger = now
                print(f"[!] ДЕТЕКТ (score={peak:.2f}) — отправляю в VK", flush=True)
                send_async(MESSAGE)

except KeyboardInterrupt:
    print("\n[*] Остановлено.")