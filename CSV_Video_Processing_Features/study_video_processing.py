import cv2
import mediapipe as mp
import numpy as np
import math
import csv
import argparse
from collections import deque


# ------------------------------------------------------------
# Mini Project: Experiment study video footage processing
# Part A: Estimate blink rate from a long study video
# Part B: Estimate approximate dimensions of eyes, face, nose, mouth
#
# Notes:
# 1. This uses MediaPipe Face Mesh landmarks.
# 2. Measurements are approximate unless a real-world scale is known.
# 3. If you know your interpupillary distance (distance between pupils),
#    you can use it to convert pixels to centimeters/mm more realistically.
# 4. If no real scale is provided, output is kept in pixels.
# ------------------------------------------------------------


# ---------------------------
# Landmark indexes
# ---------------------------
# These indexes are standard MediaPipe Face Mesh landmark ids commonly used
# for eye aspect ratio style blink detection and basic face measurements.
LEFT_EYE_HORIZONTAL = (33, 133)
RIGHT_EYE_HORIZONTAL = (362, 263)

# Vertical pairs for EAR-like blink estimation
LEFT_EYE_VERTICAL_1 = (159, 145)
LEFT_EYE_VERTICAL_2 = (158, 153)
RIGHT_EYE_VERTICAL_1 = (386, 374)
RIGHT_EYE_VERTICAL_2 = (385, 380)

# Eye corner / top-bottom rough measurement landmarks
LEFT_EYE_TOP_BOTTOM = (159, 145)
RIGHT_EYE_TOP_BOTTOM = (386, 374)

# Face width: approximate ear-to-ear substitute using wide cheek points
FACE_WIDTH = (234, 454)

# Face height: top of forehead area to chin approximation
FACE_HEIGHT = (10, 152)

# Nose width and height
NOSE_WIDTH = (129, 358)
NOSE_HEIGHT = (6, 2)

# Mouth width and height
MOUTH_WIDTH = (61, 291)
MOUTH_HEIGHT = (13, 14)

# Approximate pupil-center substitutes from eye corner midpoints
LEFT_EYE_CENTER_POINTS = [33, 133, 159, 145]
RIGHT_EYE_CENTER_POINTS = [362, 263, 386, 374]


# ---------------------------
# Utility functions
# ---------------------------
def euclidean(p1, p2):
    return math.dist(p1, p2)


def average_point(points):
    x = sum(p[0] for p in points) / len(points)
    y = sum(p[1] for p in points) / len(points)
    return (x, y)


def get_landmark_xy(landmarks, idx, width, height):
    lm = landmarks[idx]
    return (lm.x * width, lm.y * height)


def compute_eye_aspect_like_ratio(landmarks, width, height, horiz_pair, vert_pair1, vert_pair2):
    p_h1 = get_landmark_xy(landmarks, horiz_pair[0], width, height)
    p_h2 = get_landmark_xy(landmarks, horiz_pair[1], width, height)
    p_v1a = get_landmark_xy(landmarks, vert_pair1[0], width, height)
    p_v1b = get_landmark_xy(landmarks, vert_pair1[1], width, height)
    p_v2a = get_landmark_xy(landmarks, vert_pair2[0], width, height)
    p_v2b = get_landmark_xy(landmarks, vert_pair2[1], width, height)

    horizontal = euclidean(p_h1, p_h2)
    vertical1 = euclidean(p_v1a, p_v1b)
    vertical2 = euclidean(p_v2a, p_v2b)

    if horizontal == 0:
        return 0.0

    return (vertical1 + vertical2) / (2.0 * horizontal)


def compute_scale_from_ipd(landmarks, width, height, real_ipd_cm):
    left_center = average_point([get_landmark_xy(landmarks, idx, width, height) for idx in LEFT_EYE_CENTER_POINTS])
    right_center = average_point([get_landmark_xy(landmarks, idx, width, height) for idx in RIGHT_EYE_CENTER_POINTS])
    pixel_ipd = euclidean(left_center, right_center)
    if pixel_ipd <= 0:
        return None
    # cm per pixel
    return real_ipd_cm / pixel_ipd


def distance_measurement(landmarks, width, height, idx_pair, scale_cm_per_pixel=None):
    p1 = get_landmark_xy(landmarks, idx_pair[0], width, height)
    p2 = get_landmark_xy(landmarks, idx_pair[1], width, height)
    pixels = euclidean(p1, p2)
    if scale_cm_per_pixel is None:
        return pixels, "px"
    return pixels * scale_cm_per_pixel, "cm"


# ---------------------------
# Main analysis class
# ---------------------------
class StudyVideoAnalyzer:
    def __init__(self, blink_threshold=0.21, consecutive_frames=2, sample_seconds=5, real_ipd_cm=None):
        self.blink_threshold = blink_threshold
        self.consecutive_frames = consecutive_frames
        self.sample_seconds = sample_seconds
        self.real_ipd_cm = real_ipd_cm

        self.total_blinks = 0
        self.closed_counter = 0
        self.ear_history = deque(maxlen=5)
        self.measurements_over_time = []
        self.timeline_rows = []

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process_video(self, video_path, output_csv=None, preview=False):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if total_frames > 0 else 0
        sample_every_n_frames = max(1, int(fps * self.sample_seconds))

        frame_index = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.face_mesh.process(rgb)

            timestamp_sec = frame_index / fps
            left_ratio = None
            right_ratio = None
            avg_ratio = None
            blink_event = 0

            if result.multi_face_landmarks:
                landmarks = result.multi_face_landmarks[0].landmark

                left_ratio = compute_eye_aspect_like_ratio(
                    landmarks, width, height,
                    LEFT_EYE_HORIZONTAL,
                    LEFT_EYE_VERTICAL_1,
                    LEFT_EYE_VERTICAL_2,
                )
                right_ratio = compute_eye_aspect_like_ratio(
                    landmarks, width, height,
                    RIGHT_EYE_HORIZONTAL,
                    RIGHT_EYE_VERTICAL_1,
                    RIGHT_EYE_VERTICAL_2,
                )
                avg_ratio = (left_ratio + right_ratio) / 2.0
                self.ear_history.append(avg_ratio)
                smoothed_ratio = sum(self.ear_history) / len(self.ear_history)

                if smoothed_ratio < self.blink_threshold:
                    self.closed_counter += 1
                else:
                    if self.closed_counter >= self.consecutive_frames:
                        self.total_blinks += 1
                        blink_event = 1
                    self.closed_counter = 0

                if frame_index % sample_every_n_frames == 0:
                    scale = None
                    if self.real_ipd_cm is not None:
                        scale = compute_scale_from_ipd(landmarks, width, height, self.real_ipd_cm)

                    snapshot = {
                        "time_sec": round(timestamp_sec, 2),
                        "left_eye_width": distance_measurement(landmarks, width, height, LEFT_EYE_HORIZONTAL, scale),
                        "left_eye_height": distance_measurement(landmarks, width, height, LEFT_EYE_TOP_BOTTOM, scale),
                        "right_eye_width": distance_measurement(landmarks, width, height, RIGHT_EYE_HORIZONTAL, scale),
                        "right_eye_height": distance_measurement(landmarks, width, height, RIGHT_EYE_TOP_BOTTOM, scale),
                        "face_width": distance_measurement(landmarks, width, height, FACE_WIDTH, scale),
                        "face_height": distance_measurement(landmarks, width, height, FACE_HEIGHT, scale),
                        "nose_width": distance_measurement(landmarks, width, height, NOSE_WIDTH, scale),
                        "nose_height": distance_measurement(landmarks, width, height, NOSE_HEIGHT, scale),
                        "mouth_width": distance_measurement(landmarks, width, height, MOUTH_WIDTH, scale),
                        "mouth_height": distance_measurement(landmarks, width, height, MOUTH_HEIGHT, scale),
                    }
                    self.measurements_over_time.append(snapshot)

                if preview:
                    text = f"Blinks: {self.total_blinks} | Ratio: {avg_ratio:.3f}"
                    cv2.putText(frame, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Preview", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

            self.timeline_rows.append({
                "frame": frame_index,
                "time_sec": round(timestamp_sec, 3),
                "left_ratio": None if left_ratio is None else round(left_ratio, 5),
                "right_ratio": None if right_ratio is None else round(right_ratio, 5),
                "avg_ratio": None if avg_ratio is None else round(avg_ratio, 5),
                "blink_event": blink_event,
            })

            frame_index += 1

        cap.release()
        cv2.destroyAllWindows()

        total_duration = max(duration_seconds, 1e-6)
        blink_rate_per_second = self.total_blinks / total_duration
        blink_rate_per_minute = self.total_blinks / (total_duration / 60.0)

        summary = {
            "video_duration_seconds": round(duration_seconds, 2),
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "total_blinks": self.total_blinks,
            "blink_rate_per_second": round(blink_rate_per_second, 5),
            "blink_rate_per_minute": round(blink_rate_per_minute, 2),
            "average_measurements": self.compute_average_measurements(),
        }
        # ---------------------------
        # ADDITION: Estimated total blinks (for long study session)
        # ---------------------------
        study_duration_minutes = duration_seconds / 60.0
        estimated_total_blinks = blink_rate_per_minute * study_duration_minutes

        summary["estimated_total_blinks"] = round(estimated_total_blinks, 2)

        if output_csv:
            self.save_outputs(output_csv, summary)

        return summary

    def compute_average_measurements(self):
        if not self.measurements_over_time:
            return {}

        keys = [
            "left_eye_width", "left_eye_height",
            "right_eye_width", "right_eye_height",
            "face_width", "face_height",
            "nose_width", "nose_height",
            "mouth_width", "mouth_height",
        ]

        averages = {}
        for key in keys:
            vals = [entry[key][0] for entry in self.measurements_over_time if entry[key][0] is not None]
            unit = self.measurements_over_time[0][key][1]
            if vals:
                averages[key] = (round(sum(vals) / len(vals), 3), unit)

        return averages

    def save_outputs(self, output_prefix, summary):
        timeline_file = f"{output_prefix}_blink_timeline.csv"
        measures_file = f"{output_prefix}_measurements.csv"
        summary_file = f"{output_prefix}_summary.txt"

        with open(timeline_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "time_sec", "left_ratio", "right_ratio", "avg_ratio", "blink_event"])
            writer.writeheader()
            writer.writerows(self.timeline_rows)

        with open(measures_file, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "time_sec",
                "left_eye_width", "left_eye_height",
                "right_eye_width", "right_eye_height",
                "face_width", "face_height",
                "nose_width", "nose_height",
                "mouth_width", "mouth_height",
                "unit",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for row in self.measurements_over_time:
                unit = row["face_width"][1]
                out = {
                    "time_sec": row["time_sec"],
                    "left_eye_width": round(row["left_eye_width"][0], 3),
                    "left_eye_height": round(row["left_eye_height"][0], 3),
                    "right_eye_width": round(row["right_eye_width"][0], 3),
                    "right_eye_height": round(row["right_eye_height"][0], 3),
                    "face_width": round(row["face_width"][0], 3),
                    "face_height": round(row["face_height"][0], 3),
                    "nose_width": round(row["nose_width"][0], 3),
                    "nose_height": round(row["nose_height"][0], 3),
                    "mouth_width": round(row["mouth_width"][0], 3),
                    "mouth_height": round(row["mouth_height"][0], 3),
                    "unit": unit,
                }
                writer.writerow(out)

        with open(summary_file, "w", encoding="utf-8") as f:
            f.write("Study Video Analysis Summary\n")
            f.write("============================\n\n")
            for key, value in summary.items():
                if key != "average_measurements":
                    f.write(f"{key}: {value}\n")
            f.write("\nAverage Measurements:\n")
            for key, value in summary["average_measurements"].items():
                f.write(f"{key}: {value[0]} {value[1]}\n")


# ---------------------------
# Command line interface
# ---------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze videos in a local /videos folder for blink rate and facial dimensions.")
    parser.add_argument("--output_prefix", default="study_output", help="Prefix for output files")
    parser.add_argument("--blink_threshold", type=float, default=0.21)
    parser.add_argument("--consecutive_frames", type=int, default=2)
    parser.add_argument("--sample_seconds", type=int, default=5)
    parser.add_argument("--ipd_cm", type=float, default=None)
    parser.add_argument("--preview", action="store_true")

    args = parser.parse_args()

    import os

    video_folder = "videos"

    if not os.path.exists(video_folder):
        print(f"Folder '{video_folder}' not found. Create it and add your videos.")
        return

    video_files = [f for f in os.listdir(video_folder) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".lrv"))]

    if not video_files:
        print("No video files found in /videos folder.")
        return

    print(f"Found {len(video_files)} video(s): {video_files}")

    for video_file in video_files:
        video_path = os.path.join(video_folder, video_file)
        print(f"\nProcessing: {video_file}")

        analyzer = StudyVideoAnalyzer(
            blink_threshold=args.blink_threshold,
            consecutive_frames=args.consecutive_frames,
            sample_seconds=args.sample_seconds,
            real_ipd_cm=args.ipd_cm,
        )

        prefix = f"{args.output_prefix}_{os.path.splitext(video_file)[0]}"

        summary = analyzer.process_video(
            video_path=video_path,
            output_csv=prefix,
            preview=args.preview,
        )

        print("\nAnalysis Complete for:", video_file)
        print(f"Blinks: {summary['total_blinks']}")
        print(f"Blink/sec: {summary['blink_rate_per_second']}")
        print(f"Blink/min: {summary['blink_rate_per_minute']}")

        print(f"Estimated Total Blinks (from rate): {summary['estimated_total_blinks']}")
        

        print("\n--- PART B: FACIAL DIMENSIONS ---")

        avg_measures = summary["average_measurements"]

        labels = {
            "left_eye_width": "Left Eye Width",
            "left_eye_height": "Left Eye Height",
            "right_eye_width": "Right Eye Width",
            "right_eye_height": "Right Eye Height",
            "face_width": "Face Width (ear-to-ear approx)",
            "face_height": "Face Height (head-to-chin)",
            "nose_width": "Nose Width",
            "nose_height": "Nose Height",
            "mouth_width": "Mouth Width",
            "mouth_height": "Mouth Height",
        }

        for key, value in avg_measures.items():
            print(f"{labels[key]}: {value[0]} {value[1]}")


if __name__ == "__main__":
    main()
