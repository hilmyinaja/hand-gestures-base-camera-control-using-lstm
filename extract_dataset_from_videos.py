"""
Skrip Ekstraksi Fitur Landmark dari Video Mentah (.mp4) ke Dataset NumPy (.npy)
Tugas Akhir Hilmy Abid Syafi Abiyyu - NRP 5024231029 (Teknik Komputer ITS)

Fungsi:
Mengekstrak ulang 21 koordinat 3D landmark tangan (dinormalisasi terhadap wrist)
dari file video mentah di folder 'raw_videos/' menjadi array NumPy (20, 63) di 'dataset/'.
Sangat berguna jika ada file .npy yang corrupt atau jika ingin mengubah rumus normalisasi
tanpa harus merekam ulang gerakan tangan di depan kamera.
"""

import cv2
import os
import glob
import numpy as np
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

TARGET_FPS = 20
SEQUENCE_LENGTH = 20
DATASET_DIR = "dataset"
RAW_VIDEOS_DIR = "raw_videos"
MODEL_PATH = "hand_landmarker.task"

def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        print(f"[Info] Mengunduh model {MODEL_PATH}...")
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
        urllib.request.urlretrieve(url, MODEL_PATH)
        print("[Info] Model berhasil diunduh!")

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

def process_single_video(detector, video_path, output_npy_path):
    """Membaca video .mp4 frame-by-frame dan menyimpan array (20, 63) ke .npy."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[Error] Tidak bisa membuka video: {video_path}")
        return False

    frames_features = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # MediaPipe video mode membutuhkan timestamp dalam milidetik
        timestamp_ms = int(frame_idx * (1000.0 / TARGET_FPS))
        result = detector.detect_for_video(mp_image, timestamp_ms)

        if result and result.hand_landmarks:
            features = extract_normalized_landmarks(result.hand_landmarks[0])
            frames_features.append(features)
        else:
            # Fallback jika ada 1 frame tangan terlewat: duplikasi frame sebelumnya atau koordinat 0
            if frames_features:
                frames_features.append(frames_features[-1])
            else:
                frames_features.append(np.zeros(63, dtype=np.float32))

        frame_idx += 1
        if frame_idx >= SEQUENCE_LENGTH:
            break

    cap.release()

    if len(frames_features) < SEQUENCE_LENGTH:
        # Pad jika kurang dari 20 frame
        while len(frames_features) < SEQUENCE_LENGTH:
            frames_features.append(frames_features[-1] if frames_features else np.zeros(63, dtype=np.float32))

    data_array = np.array(frames_features[:SEQUENCE_LENGTH], dtype=np.float32)
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)
    np.save(output_npy_path, data_array)
    return True

def main():
    print("=================================================================")
    print("   Ekstraktor Landmark dari Video Mentah (.mp4) ke .npy          ")
    print("=================================================================")

    ensure_model_exists()

    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.5,
        running_mode=vision.RunningMode.VIDEO
    )
    detector = vision.HandLandmarker.create_from_options(options)

    if not os.path.exists(RAW_VIDEOS_DIR):
        print(f"[Info] Folder '{RAW_VIDEOS_DIR}' belum ada. Rekam data terlebih dahulu dengan data_collector.py.")
        return

    classes = [d for d in os.listdir(RAW_VIDEOS_DIR) if os.path.isdir(os.path.join(RAW_VIDEOS_DIR, d))]
    total_processed = 0

    for c in sorted(classes):
        video_files = sorted(glob.glob(os.path.join(RAW_VIDEOS_DIR, c, "*.mp4")))
        print(f"\n📂 Memproses kelas: '{c}' ({len(video_files)} video klip)")

        for v_path in video_files:
            file_base = os.path.basename(v_path).replace(".mp4", ".npy")
            npy_path = os.path.join(DATASET_DIR, c, file_base)

            success = process_single_video(detector, v_path, npy_path)
            if success:
                total_processed += 1

    detector.close()
    print(f"\n✅ Selesai! Sebanyak {total_processed} file sekuens .npy berhasil diekstrak dan disimpan di '{DATASET_DIR}'.")

if __name__ == "__main__":
    main()
