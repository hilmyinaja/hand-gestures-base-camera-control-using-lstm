"""
Sistem Kontrol Parameter Kamera Berbasis Hand Gesture Real-Time
Menggunakan MediaPipe Hands + TFLite Stacked LSTM pada Perangkat Edge
Hilmy Abid Syafi Abiyyu (NRP 5024231029) - Teknik Komputer FTEIC ITS
"""

import cv2
import time
import os
import json
import collections
import platform
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# --- Konfigurasi Sesuai Bab 3 Metodologi ---
TARGET_FPS = 20
SEQUENCE_LENGTH = 20           # 20 frame temporal sliding window
FRAME_INTERVAL = 1.0 / TARGET_FPS  # 50 ms antar frame

CONFIDENCE_THRESHOLD = 0.80    # Ambang batas keyakinan (Subbab 3.4.1 Poin 1)
DISCRETE_COOLDOWN = 1.2        # Jeda aksi diskrit: foto, ganti mode, lock (detik)
SCROLL_COOLDOWN = 0.3          # Jeda scroll kontinu: putar nilai & zoom (300 ms)
SAFETY_TIMEOUT_SECONDS = 5.0   # Batas waktu buka kunci rekaman sebelum auto re-lock

MODEL_TFLITE_PATH = "model_gesture.tflite"
CLASSES_JSON_PATH = "gesture_classes.json"
MEDIAPIPE_MODEL_PATH = "hand_landmarker.task"

# Topology 21 Landmark Tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

# Tabel Parameter Kamera (Simulasi Terintegrasi)
CAMERA_MODES = ["ISO", "Aperture", "Shutter", "Zoom"]
PARAM_TABLES = {
    "ISO": ["100", "200", "400", "800", "1600", "3200", "6400"],
    "Aperture": ["f/1.8", "f/2.8", "f/4.0", "f/5.6", "f/8.0", "f/11", "f/16"],
    "Shutter": ["1/30s", "1/60s", "1/125s", "1/250s", "1/500s", "1/1000s"],
    "Zoom": ["1.0x", "1.2x", "1.5x", "1.8x", "2.0x", "2.5x", "3.0x"]
}

def load_classes():
    if os.path.exists(CLASSES_JSON_PATH):
        with open(CLASSES_JSON_PATH, "r") as f:
            return json.load(f)
    return [
        "rotate_cw", "rotate_ccw", "next_mode", "lock_unlock",
        "zoom_in", "zoom_out", "snap_photo", "record_video", "idle"
    ]

def extract_normalized_landmarks(hand_landmarks):
    """Mengekstraksi 21 koordinat 3D dinormalisasi relatif terhadap wrist (landmark 0)."""
    wrist = hand_landmarks[0]
    features = []
    for lm in hand_landmarks:
        features.extend([
            lm.x - wrist.x,
            lm.y - wrist.y,
            lm.z - wrist.z
        ])
    return np.array(features, dtype=np.float32)

def open_camera():
    is_windows = (platform.system() == "Windows")
    backend = cv2.CAP_DSHOW if is_windows else cv2.CAP_V4L2
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def main():
    print("=================================================================")
    print("   SISTEM KONTROL GESTUR KAMERA REAL-TIME (FASE 5)               ")
    print("=================================================================")

    # 1. Muat Label Kelas
    classes = load_classes()
    print(f"✔ Kelas Gestur ({len(classes)}): {classes}")

    # 2. Inisialisasi TFLite Interpreter
    if not os.path.exists(MODEL_TFLITE_PATH):
        print(f"[Error] Model '{MODEL_TFLITE_PATH}' belum ada! Jalankan 'py train_lstm.py' terlebih dahulu.")
        return

    interpreter = tf.lite.Interpreter(model_path=MODEL_TFLITE_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✔ Model TFLite berhasil dimuat dan dialokasikan ke memori.")

    # 3. Inisialisasi Detektor MediaPipe Hands
    if not os.path.exists(MEDIAPIPE_MODEL_PATH):
        print(f"[Error] Model '{MEDIAPIPE_MODEL_PATH}' tidak ditemukan!")
        return

    base_options = python.BaseOptions(model_asset_path=MEDIAPIPE_MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # 4. Buka Kamera Masukan
    cap = open_camera()
    if not cap.isOpened():
        print("[Error] Webcam tidak terdeteksi!")
        return
    print("✔ Webcam aktif! Jendela antarmuka visual terbuka.")

    # --- Parameter State Machine Sistem Kamera ---
    current_mode_idx = 0       # 0: ISO, 1: Aperture, 2: Shutter, 3: Zoom
    param_indices = {
        "ISO": 1,        # 200
        "Aperture": 1,   # f/2.8
        "Shutter": 2,    # 1/125s
        "Zoom": 0        # 1.0x
    }

    is_locked = False                  # Status Kunci Sistem
    is_recording = False               # Status Rekam Video
    unlock_for_stop_pending = False    # Kandidat stop video setelah Fist Unlock
    unlock_pending_start_time = 0      # Timer 5 detik fail-safe

    # Sliding Window Buffer & Timers
    sliding_window = collections.deque(maxlen=SEQUENCE_LENGTH)
    last_sample_time = 0
    last_trigger_time = 0
    last_action_msg = "Sistem Siap (Standby)"
    last_action_color = (0, 255, 128)
    shutter_flash_time = 0

    mirror_view = True
    start_time = time.time()
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        if mirror_view:
            frame = cv2.flip(frame, 1)

        h, w, _ = frame.shape
        now = time.time()

        # Hitung FPS
        fps = 1.0 / (now - prev_time) if (now - prev_time) > 0 else 0
        prev_time = now

        # Ekstraksi MediaPipe Hands
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int((now - start_time) * 1000)
        result = detector.detect_for_video(mp_image, timestamp_ms)

        hand_present = bool(result and result.hand_landmarks)

        # Rate-limiter 20 FPS ke sliding window buffer
        if (now - last_sample_time) >= FRAME_INTERVAL:
            if hand_present:
                lm_norm = extract_normalized_landmarks(result.hand_landmarks[0])
                sliding_window.append(lm_norm)
            else:
                # Jika tangan keluar dari frame, perlahan geser antrean
                if len(sliding_window) > 0:
                    sliding_window.popleft()
            last_sample_time = now

        # Gambar Landmark Tangan
        if hand_present:
            hand_landmarks = result.hand_landmarks[0]
            coords = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            for s, e in HAND_CONNECTIONS:
                cv2.line(frame, coords[s], coords[e], (0, 255, 128), 2)
            for idx, pt in enumerate(coords):
                color = (0, 0, 255) if idx == 0 else (255, 100, 0)
                size = 6 if idx == 0 else 4
                cv2.circle(frame, pt, size, color, -1)

        # --- Evaluasi Inferensi TFLite ---
        detected_class = "idle"
        detected_confidence = 0.0
        all_probs = [0.0] * len(classes)

        if len(sliding_window) == SEQUENCE_LENGTH:
            input_tensor = np.expand_dims(np.array(sliding_window, dtype=np.float32), axis=0)
            interpreter.set_tensor(input_details[0]['index'], input_tensor)
            interpreter.invoke()
            output_tensor = interpreter.get_tensor(output_details[0]['index'])[0]

            all_probs = output_tensor
            max_idx = int(np.argmax(output_tensor))
            detected_confidence = float(output_tensor[max_idx])
            detected_class = classes[max_idx]

        # --- 2-Step Video Safety Guard Fail-Safe Timeout ---
        if is_recording and unlock_for_stop_pending:
            elapsed_unlock = now - unlock_pending_start_time
            if elapsed_unlock > SAFETY_TIMEOUT_SECONDS:
                # Timeout 5 detik habis! Kunci otomatis kembali
                unlock_for_stop_pending = False
                is_locked = True
                last_action_msg = "Timeout 5s! Sistem terkunci kembali"
                last_action_color = (0, 0, 255)

        # --- Logika Eksekusi Perintah & State Machine ---
        # Terapkan ambang batas keyakinan (Confidence Threshold >= 0.80)
        is_confident = (detected_confidence >= CONFIDENCE_THRESHOLD) and (detected_class != "idle")

        if is_confident:
            active_param_name = CAMERA_MODES[current_mode_idx]
            time_since_last_trigger = now - last_trigger_time

            # 1. GESTUR KUNCI / BUKA KUNCI (Fist ✊)
            if detected_class == "lock_unlock":
                if time_since_last_trigger >= DISCRETE_COOLDOWN:
                    if is_recording:
                        if not unlock_for_stop_pending:
                            # Buka kunci untuk persiapan stop rekaman
                            unlock_for_stop_pending = True
                            is_locked = False
                            unlock_pending_start_time = now
                            last_action_msg = "KUNCI DIBUKA! Lakukan ✌️ dlm 5s utk Stop Video"
                            last_action_color = (0, 255, 255)
                    else:
                        is_locked = not is_locked
                        status_str = "TERKUNCI 🔒" if is_locked else "TERBUKA 🔓"
                        last_action_msg = f"Sistem {status_str}"
                        last_action_color = (0, 0, 255) if is_locked else (0, 255, 0)
                    last_trigger_time = now

            # 2. GESTUR REKAM VIDEO (Peace ✌️)
            elif detected_class == "record_video":
                if time_since_last_trigger >= DISCRETE_COOLDOWN:
                    if not is_recording:
                        # Mulai Rekam -> AUTO LOCK seketika!
                        if not is_locked:
                            is_recording = True
                            is_locked = True
                            unlock_for_stop_pending = False
                            last_action_msg = "🔴 REKAM VIDEO DIMULAI (AUTO-LOCKED)"
                            last_action_color = (0, 0, 255)
                            last_trigger_time = now
                    else:
                        # Hentikan Rekam (Hanya boleh jika kunci sudah dibuka dengan Fist ✊)
                        if not is_locked and unlock_for_stop_pending:
                            is_recording = False
                            unlock_for_stop_pending = False
                            is_locked = False
                            last_action_msg = "⏹️ REKAM VIDEO SELESAI & TERSIMPAN"
                            last_action_color = (0, 255, 0)
                            last_trigger_time = now
                        else:
                            # Abaikan jika masih terkunci
                            pass

            # 3. GESTUR LAINNYA (Hanya dieksekusi jika sistem UNLOCKED)
            elif not is_locked and not is_recording:
                # A. Ganti Parameter Aktif (Swipe Up)
                if detected_class == "next_mode":
                    if time_since_last_trigger >= DISCRETE_COOLDOWN:
                        current_mode_idx = (current_mode_idx + 1) % len(CAMERA_MODES)
                        new_mode = CAMERA_MODES[current_mode_idx]
                        last_action_msg = f"Mode Aktif: {new_mode} ({param_indices[new_mode]})"
                        last_action_color = (0, 255, 255)
                        last_trigger_time = now

                # B. Jepret Foto (Snap Photo 🤏)
                elif detected_class == "snap_photo":
                    if time_since_last_trigger >= DISCRETE_COOLDOWN:
                        shutter_flash_time = now
                        last_action_msg = "📸 JEPRET FOTO! Shutter Released"
                        last_action_color = (0, 255, 0)
                        last_trigger_time = now

                # C. Putar Searah Jarum Jam (Rotate CW - Naikkan Nilai)
                elif detected_class == "rotate_cw":
                    if time_since_last_trigger >= SCROLL_COOLDOWN:
                        table = PARAM_TABLES[active_param_name]
                        if param_indices[active_param_name] < len(table) - 1:
                            param_indices[active_param_name] += 1
                        val = table[param_indices[active_param_name]]
                        last_action_msg = f"{active_param_name} Naik (+1) ➔ {val}"
                        last_action_color = (0, 255, 128)
                        last_trigger_time = now

                # D. Putar Lawan Jarum Jam (Rotate CCW - Turunkan Nilai)
                elif detected_class == "rotate_ccw":
                    if time_since_last_trigger >= SCROLL_COOLDOWN:
                        table = PARAM_TABLES[active_param_name]
                        if param_indices[active_param_name] > 0:
                            param_indices[active_param_name] -= 1
                        val = table[param_indices[active_param_name]]
                        last_action_msg = f"{active_param_name} Turun (-1) ➔ {val}"
                        last_action_color = (0, 200, 255)
                        last_trigger_time = now

                # E. Zoom In (Pinch Out)
                elif detected_class == "zoom_in":
                    if time_since_last_trigger >= SCROLL_COOLDOWN:
                        z_table = PARAM_TABLES["Zoom"]
                        if param_indices["Zoom"] < len(z_table) - 1:
                            param_indices["Zoom"] += 1
                        last_action_msg = f"Zoom In (+): {z_table[param_indices['Zoom']]}"
                        last_action_color = (255, 128, 0)
                        last_trigger_time = now

                # F. Zoom Out (Pinch In)
                elif detected_class == "zoom_out":
                    if time_since_last_trigger >= SCROLL_COOLDOWN:
                        z_table = PARAM_TABLES["Zoom"]
                        if param_indices["Zoom"] > 0:
                            param_indices["Zoom"] -= 1
                        last_action_msg = f"Zoom Out (-): {z_table[param_indices['Zoom']]}"
                        last_action_color = (255, 128, 0)
                        last_trigger_time = now

        # --- Efek Visual Shutter Flash Foto ---
        if now - shutter_flash_time < 0.12:
            frame[:, :] = 255  # Layar putih sekejap seperti flash kamera

        # --- TAMPILAN DASHBOARD HUD KAMERA (Overlay) ---
        # 1. Kotak Parameter Kamera di Atas
        hud_bg_y = 65
        cv2.rectangle(frame, (10, 10), (w - 10, hud_bg_y), (25, 25, 25), -1)

        col_w = (w - 20) // 4
        for idx, mode_name in enumerate(CAMERA_MODES):
            bx = 10 + idx * col_w
            is_active = (idx == current_mode_idx)
            val = PARAM_TABLES[mode_name][param_indices[mode_name]]

            # Warna kartu parameter
            card_border = (0, 255, 255) if is_active else (80, 80, 80)
            cv2.rectangle(frame, (bx + 3, 14), (bx + col_w - 3, hud_bg_y - 4), card_border, 1 if not is_active else 2)

            label_txt = f"[{mode_name}]" if is_active else mode_name
            cv2.putText(frame, label_txt, (bx + 8, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (0, 255, 255) if is_active else (180, 180, 180), 1)
            cv2.putText(frame, val, (bx + 8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                        (255, 255, 255), 2 if is_active else 1)

        # 2. Status Bar Keamanan (Lock / Video Safety)
        status_box_y = 75
        cv2.rectangle(frame, (10, status_box_y), (w - 10, status_box_y + 40), (20, 20, 20), -1)

        if is_recording:
            if unlock_for_stop_pending:
                rem = max(0.0, SAFETY_TIMEOUT_SECONDS - (now - unlock_pending_start_time))
                lock_text = f"SIAP STOP ({rem:.1f}s) - Gestur ✌️"
                lock_color = (0, 255, 255)
            else:
                lock_text = "MEREKAM VIDEO (AUTO-LOCKED) 🔒"
                lock_color = (0, 0, 255)
        else:
            lock_text = "TERKUNCI (LOCKED) 🔒" if is_locked else "TERBUKA (UNLOCKED) 🔓"
            lock_color = (0, 0, 255) if is_locked else (0, 255, 0)

        cv2.putText(frame, lock_text, (20, status_box_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lock_color, 2)

        # Feedback Aksi Terakhir di sebelah kanan status bar
        cv2.putText(frame, last_action_msg, (280, status_box_y + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, last_action_color, 1)

        # 3. Bar Probabilitas Deteksi Terkini di Bawah
        bot_y = h - 45
        cv2.rectangle(frame, (10, bot_y), (w - 10, h - 10), (25, 25, 25), -1)

        det_color = (0, 255, 128) if detected_confidence >= CONFIDENCE_THRESHOLD else (140, 140, 140)
        cv2.putText(frame, f"AI: {detected_class.upper()} ({detected_confidence*100:.1f}%)",
                    (20, bot_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, det_color, 2)

        # Bar Visual Confidence
        bar_x = 280
        bar_w = int(detected_confidence * (w - bar_x - 30))
        cv2.rectangle(frame, (bar_x, bot_y + 12), (bar_x + bar_w, bot_y + 23), det_color, -1)
        cv2.rectangle(frame, (bar_x, bot_y + 12), (w - 30, bot_y + 23), (80, 80, 80), 1)

        # 4. Panel Mini 9 Probabilitas Kelas (di kanan)
        menu_y = 135
        for i, c_name in enumerate(classes):
            p = all_probs[i] if i < len(all_probs) else 0.0
            is_top = (c_name == detected_class and p >= CONFIDENCE_THRESHOLD)
            c_color = (0, 255, 255) if is_top else (150, 150, 150)
            cv2.putText(frame, f"{c_name[:11]:<11}: {p*100:4.1f}%", (w - 175, menu_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, c_color, 1)
            menu_y += 16

        cv2.imshow("Kontrol Gestur Kamera - Tugas Akhir Hilmy", frame)

        # Penanganan Keyboard
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('m') or key == ord('M'):
            mirror_view = not mirror_view
        elif key == ord('l') or key == ord('L'):
            is_locked = not is_locked
            last_action_msg = f"Manual Toggle: {'TERKUNCI' if is_locked else 'TERBUKA'}"
            last_action_color = (0, 0, 255) if is_locked else (0, 255, 0)
        elif key == ord('r') or key == ord('R'):
            is_recording = not is_recording
            last_action_msg = f"Manual Video: {'REC' if is_recording else 'STOP'}"

    cap.release()
    cv2.destroyAllWindows()
    try:
        detector.close()
    except:
        pass
    print("\nSistem kontrol gestur dimatikan.")

if __name__ == "__main__":
    main()
