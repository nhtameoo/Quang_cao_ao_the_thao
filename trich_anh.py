"""
extract_frames.py
Trích frame từ video để train YOLO billboard.
Lưu vào: C:\\Users\\nhta4\\Desktop\\Image
"""

import os
import cv2
import numpy as np

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
VIDEO_PATH = r"C:\Users\nhta4\Desktop\video.mp4"
OUTPUT_DIR = r"C:\Users\nhta4\Desktop\Image"

# Số frame muốn lưu (khuyến nghị 300-500)
TARGET_FRAMES = 400

# Bỏ qua N frame đầu (nếu video có intro/quảng cáo)
SKIP_START = 0

# Bỏ qua N frame cuối (nếu có outro)
SKIP_END = 0

# Lọc frame bị mờ (Laplacian variance thấp = mờ)
MIN_SHARPNESS = 100.0

# Số frame tối thiểu giữa 2 frame lưu (tránh lưu frame quá giống nhau)
MIN_FRAME_GAP = 5


# ==============================================================================
# HÀM HỖ TRỢ
# ==============================================================================
def sharpness_score(frame):
    """Đo độ nét của frame bằng Laplacian variance."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def frame_similarity(frame1, frame2, size=(160, 90)):
    """Đo độ giống nhau giữa 2 frame (0.0 = khác hoàn toàn, 1.0 = giống hệt)."""
    a = cv2.resize(frame1, size).astype(np.float32)
    b = cv2.resize(frame2, size).astype(np.float32)
    diff = np.abs(a - b).mean()
    return 1.0 - min(diff / 128.0, 1.0)


# ==============================================================================
# MAIN
# ==============================================================================
def extract_frames():
    if not os.path.exists(VIDEO_PATH):
        print(f"LỖI: Không tìm thấy video: {VIDEO_PATH}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"LỖI: Không mở được video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 70)
    print(f"Video     : {VIDEO_PATH}")
    print(f"Tổng frame: {total_frames}")
    print(f"FPS       : {fps:.2f}")
    print(f"Kích thước: {w} x {h}")
    print(f"Thời lượng: {total_frames / fps:.1f} giây")
    print(f"Output    : {OUTPUT_DIR}")
    print(f"Mục tiêu  : {TARGET_FRAMES} frame")
    print("=" * 70)

    # Khoảng cách frame nếu lấy đều
    start = SKIP_START
    end = max(start + 1, total_frames - SKIP_END)
    step = max(1, (end - start) // TARGET_FRAMES)

    print(f"\nBắt đầu trích frame (bước = {step})...\n")

    saved = 0
    skipped_blur = 0
    skipped_similar = 0
    last_saved_frame = None
    last_saved_idx = -MIN_FRAME_GAP - 1

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Bỏ qua đầu/cuối
        if frame_idx < start or frame_idx >= end:
            frame_idx += 1
            continue

        # Bỏ qua nếu không phải bước nhảy
        if (frame_idx - start) % step != 0:
            frame_idx += 1
            continue

        # Kiểm tra độ nét
        sharp = sharpness_score(frame)
        if sharp < MIN_SHARPNESS:
            skipped_blur += 1
            frame_idx += 1
            continue

        # Kiểm tra giống frame trước
        if (last_saved_frame is not None
                and frame_idx - last_saved_idx < MIN_FRAME_GAP * 10):
            sim = frame_similarity(frame, last_saved_frame)
            if sim > 0.98:
                skipped_similar += 1
                frame_idx += 1
                continue

        # Lưu frame
        filename = f"frame_{saved:05d}.jpg"
        filepath = os.path.join(OUTPUT_DIR, filename)
        cv2.imwrite(filepath, frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])

        saved += 1
        last_saved_frame = frame.copy()
        last_saved_idx = frame_idx

        # In tiến độ
        if saved % 20 == 0:
            print(f"  Đã lưu {saved} frame (frame gốc #{frame_idx}, sharpness={sharp:.1f})")

        # Đủ số lượng → dừng
        if saved >= TARGET_FRAMES:
            break

        frame_idx += 1

    cap.release()

    print("\n" + "=" * 70)
    print(f"HOÀN THÀNH")
    print(f"  Đã lưu         : {saved} frame")
    print(f"  Bỏ qua (mờ)    : {skipped_blur} frame")
    print(f"  Bỏ qua (giống) : {skipped_similar} frame")
    print(f"  Thư mục        : {OUTPUT_DIR}")
    print("=" * 70)
    print("\nBước tiếp theo:")
    print("  1. Cài LabelImg:  pip install labelImg")
    print("  2. Chạy:          labelImg")
    print(f"  3. Mở thư mục:    {OUTPUT_DIR}")
    print("  4. Vẽ bounding box quanh biển quảng cáo, class = billboard")
    print("  5. Lưu dạng YOLO format")


if __name__ == "__main__":
    extract_frames()