import os
import cv2
import numpy as np
import mediapipe as mp
import pandas as pd
from tqdm import tqdm
import argparse
from collections import defaultdict


def calculate_geometric_features(landmarks, img_w, img_h):
    if not isinstance(landmarks, np.ndarray):
        coords = np.array([(lm.x * img_w, lm.y * img_h) for lm in landmarks])
    else:
        coords = landmarks

    ref_dist = np.linalg.norm(coords[133] - coords[362])
    if ref_dist < 1e-6:
        return None

    features = []

    # Mata: aspect ratio
    def eye_aspect_ratio(eye_coords):
        v1 = np.linalg.norm(eye_coords[1] - eye_coords[5])
        v2 = np.linalg.norm(eye_coords[2] - eye_coords[4])
        h = np.linalg.norm(eye_coords[0] - eye_coords[3])
        return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0

    features.append(eye_aspect_ratio(coords[[33, 160, 158, 133, 153, 144]]))
    features.append(eye_aspect_ratio(coords[[362, 385, 387, 263, 373, 380]]))
    left_corner = coords[61]
    right_corner = coords[291]
    top_outer = coords[0]
    bottom_outer = coords[17]
    top_mid = coords[37]
    bottom_mid = coords[267]
    mouth_width = np.linalg.norm(left_corner - right_corner)
    mouth_height = np.linalg.norm(top_outer - bottom_outer)
    mouth_curve = (np.linalg.norm(left_corner - top_mid) +
                   np.linalg.norm(right_corner - top_mid) +
                   np.linalg.norm(left_corner - bottom_mid) +
                   np.linalg.norm(right_corner - bottom_mid)) / 4
    features.extend([
        mouth_width / ref_dist if ref_dist > 1e-6 else 0.0,
        mouth_height / mouth_width if mouth_width > 1e-6 else 0.0,
        mouth_curve / mouth_width if mouth_width > 1e-6 else 0.0,
        mouth_height / ref_dist if ref_dist > 1e-6 else 0.0
    ])
    left_brow = coords[[70, 63, 105, 66, 107]]
    right_brow = coords[[336, 296, 334, 293, 300]]

    def brow_angle(p):
        return np.degrees(np.arctan2(p[-1][1] - p[0][1], p[-1][0] - p[0][0]))
    features.append(brow_angle(left_brow))
    features.append(brow_angle(right_brow))
    features.extend([
        np.linalg.norm(coords[107] - coords[336]) / ref_dist,
        np.linalg.norm(coords[152] - coords[17]) / ref_dist,
        np.linalg.norm(coords[234] - coords[454]) / ref_dist,
        np.linalg.norm(coords[168] - coords[8]) / ref_dist
    ])
    return np.array(features)


def _process_image(frame, face_mesh):
    if frame is None:
        return None
    results = face_mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        features = calculate_geometric_features(
            results.multi_face_landmarks[0].landmark, frame.shape[1], frame.shape[0]
        )
        return features
    return None


def run_global_baseline_creation(source_dir, output_dir):
    print("🚀 Memulai proses pembuatan baseline netral global...")
    neutral_dir = os.path.join(source_dir, 'neutral')
    if not os.path.isdir(neutral_dir):
        print(f"❌ Error: Direktori '{neutral_dir}' tidak ditemukan.")
        return
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    all_neutral_features = []
    image_files = [f for f in os.listdir(neutral_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"📊 Ditemukan {len(image_files)} gambar di folder 'neutral'. Memproses...")
    for img_name in tqdm(image_files, desc="Processing Neutral Images"):
        img_path = os.path.join(neutral_dir, img_name)
        frame = cv2.imread(img_path)
        original_features = _process_image(frame, mp_face_mesh)
        if original_features is not None:
            all_neutral_features.append(original_features)
        flipped_frame = cv2.flip(frame, 1)
        flipped_features = _process_image(flipped_frame, mp_face_mesh)
        if flipped_features is not None:
            all_neutral_features.append(flipped_features)
    mp_face_mesh.close()
    if not all_neutral_features:
        print("❌ Tidak ada fitur netral yang berhasil diekstrak. Proses dibatalkan.")
        return
    global_baseline = np.mean(all_neutral_features, axis=0)
    output_path = os.path.join(output_dir, 'global_neutral_baseline.npy')
    np.save(output_path, global_baseline)
    print("\n" + "="*50)
    print(f"✅ Global baseline berhasil dibuat di: {output_path}")
    print(f"   Total sampel fitur yang diproses: {len(all_neutral_features)}")
    print(f"   Dimensi vektor baseline: {global_baseline.shape}")
    print("="*50)


def _get_subject_file_map(source_dir):
    subjects = defaultdict(lambda: defaultdict(list))
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            try:
                subject_id = img_name.split('_')[0]
                subjects[subject_id][class_name].append(os.path.join(class_path, img_name))
            except IndexError:
                print(f"Peringatan: Melewati file dengan format nama yang tidak valid: {img_name}")
    return subjects


def _calculate_subject_baselines(subjects, face_mesh):
    neutral_baselines = {}
    print("📊 Menghitung baseline netral untuk setiap subjek...")
    for subject_id, emotions in tqdm(subjects.items(), desc="Calculating Subject Baselines"):
        neutral_features = []
        if 'neutral' in emotions:
            for img_path in emotions['neutral']:
                features = _process_image(cv2.imread(img_path), face_mesh)
                if features is not None:
                    neutral_features.append(features)
        if neutral_features:
            neutral_baselines[subject_id] = np.mean(neutral_features, axis=0)
    return neutral_baselines


def run_delta_feature_extraction(source_dir, output_dir):
    mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    subjects = _get_subject_file_map(source_dir)
    neutral_baselines = _calculate_subject_baselines(subjects, mp_face_mesh)
    all_delta_features = []
    class_dirs = [d for d in sorted(os.listdir(source_dir)) if os.path.isdir(os.path.join(source_dir, d))]
    class_to_idx = {name: i for i, name in enumerate(class_dirs)}
    for subject_id, emotions in tqdm(subjects.items(), desc="Calculating Delta Features"):
        if subject_id not in neutral_baselines:
            print(f"Peringatan: Melewati subjek '{subject_id}' karena tidak memiliki baseline netral.")
            continue
        baseline = neutral_baselines[subject_id]
        for emotion_name, img_paths in emotions.items():
            label = class_to_idx[emotion_name]
            for img_path in img_paths:
                current_features = _process_image(cv2.imread(img_path), mp_face_mesh)
                if current_features is not None:
                    delta_features = current_features - baseline
                    all_delta_features.append(np.append(delta_features, label))
    mp_face_mesh.close()
    if not all_delta_features:
        print("❌ Tidak ada delta fitur yang berhasil diekstrak. Proses dibatalkan.")
        return
    df = pd.DataFrame(all_delta_features)
    num_features = df.shape[1] - 1
    df.columns = [f'delta_feature_{i}' for i in range(num_features)] + ['label']
    output_csv_path = os.path.join(output_dir, "delta_emotion_features.csv")
    output_map_path = os.path.join(output_dir, "map_label.py")
    df.to_csv(output_csv_path, index=False)
    with open(output_map_path, "w") as f:
        f.write("# File ini dibuat secara otomatis oleh preprocess_master.py\n")
        f.write(f"CLASS_NAMES = {class_dirs}\n")
    print(f"\n✅ Preprocessing Delta Fitur selesai. {len(all_delta_features)} sampel diproses.")
    print(f"✅ Delta fitur berhasil disimpan ke: {output_csv_path}")
    print(f"✅ Mapping label berhasil disimpan ke: {output_map_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Skrip Preprocessing untuk membuat baseline global DAN fitur delta secara sekuensial.",
        formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument("--input", type=str, required=True,
                        help="Path ke direktori data gambar sumber (mis: ./Data.facial).")

    parser.add_argument("--output_dir", type=str, default="./runs",
                        help="Direktori untuk menyimpan semua file output.")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print("\n--- LANGKAH 1: MENJALANKAN PEMBUATAN BASELINE GLOBAL ---")
    run_global_baseline_creation(args.input, args.output_dir)
    print("\n--- LANGKAH 2: MENJALANKAN EKSTRAKSI FITUR DELTA ---")
    os.makedirs(args.output_dir, exist_ok=True)
    run_delta_feature_extraction(args.input, args.output_dir)
    print("\n🎉 Semua proses preprocessing telah selesai.")


if __name__ == "__main__":
    main()
