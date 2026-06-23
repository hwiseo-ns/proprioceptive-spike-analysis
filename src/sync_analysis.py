"""
sync_analysis

Syncronize FR(Firing Rate) with limb kinetics
1. Calculate FR from spike2 spike wavemarks
2. Estimate time offset
3. Find correlation kinetics best fit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import stats
import cv2

def get_video_properties(video_path="./data/20250710_173046.mp4"):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.release()
    
    return fps, total_frames

video_fps, total_frames = get_video_properties()
timescale = np.arange(1/video_fps, total_frames/video_fps, 1/video_fps)
start_sec = 200
end_sec = 300
offset_sec = -1.367  # Adjust this value to synchronize the two signals

def normalize_array(arr, mode="min-max"):
    if mode == "z-score":
        arr_mean = np.mean(arr)
        arr_std = np.std(arr, ddof=0)
        arr_normalized = (arr - arr_mean) / arr_std if arr_std != 0 else arr - arr_mean
    elif mode == "min-max":
        arr_min = np.min(arr)
        arr_max = np.max(arr)
        arr_normalized = (arr - arr_min) / (arr_max - arr_min) if arr_max != arr_min else np.zeros_like(arr)
    return arr_normalized

def calculate_FR(bandwidth=0.001):
    with open("./data/spike_times.txt", "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]

    f_data = np.array(data)

    # get FR with Kernel Density Estimation
    kde = gaussian_kde(f_data)
    kde.set_bandwidth(bw_method=bandwidth)

    y_density = kde(timescale) * len(f_data)

    # return raw density; normalization will be applied after slicing
    return y_density

def calculate_kinetics(csv_path="./data/20250710_173046DLC_Resnet50_rabbit_analysisJun22shuffle1_snapshot_200.csv"):
    # Overview of Kinetics
    file = csv_path
    df = pd.read_csv(file, header=[0, 1, 2], index_col=0)

    # Extract knee and ankle header
    try:
        knee_x = df.xs(('knee', 'x'), level=(1, 2), axis=1).iloc[:, 0]
        knee_y = df.xs(('knee', 'y'), level=(1, 2), axis=1).iloc[:, 0]
        ankle_x = df.xs(('ankle', 'x'), level=(1, 2), axis=1).iloc[:, 0]
        ankle_y = df.xs(('ankle', 'y'), level=(1, 2), axis=1).iloc[:, 0]
    except KeyError as e:
        print(f"Column could not find Error: {e}")

    # Seg length
    seg_length = np.sqrt((ankle_x - knee_x)**2 + (ankle_y - knee_y)**2)

    # return raw segment lengths; normalization will be applied after slicing
    return seg_length

def plot_FR_and_kinetics(offset_sec=0.0):
    mask = (timescale >= start_sec) & (timescale <= end_sec)
    mask_offset = (timescale >= start_sec + offset_sec) & (timescale <= end_sec + offset_sec)
    y_density = calculate_FR()
    seg_length = calculate_kinetics()
    t_slice = timescale[mask]
    y_slice = normalize_array(y_density[mask])
    seg_slice = normalize_array(seg_length[mask_offset])
    
    pearson_r, p_val_p = stats.pearsonr(y_slice, seg_slice)
    spearman_r, p_val_s = stats.spearmanr(y_slice, seg_slice)

    print(f"Pearson r: {pearson_r:.4f} (p-value: {p_val_p})")
    print(f"Spearman r: {spearman_r:.4f} (p-value: {p_val_s})")

    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True, facecolor="#11161d")

    ax1.plot(t_slice, y_slice, color="#5897e5", linewidth=1.5, label="Frequency (KDE)")
    ax1.fill_between(t_slice, y_slice, color="#20334e", alpha=0.5)
    ax1.set_title("Firing Rate", fontsize=13, weight="bold", color="#e0e0e0")
    ax1.set_ylabel("Frequency (#/s)", fontsize=11, color="#8a95a5")
    ax1.grid(True, color="#252e3b", linestyle="-", linewidth=0.5)
    ax1.set_xlim(t_slice.min(), t_slice.max())

    ax2.plot(t_slice, seg_slice, color="#34b360", linewidth=1.2)
    ax2.fill_between(t_slice, seg_slice, color="#153d23", alpha=0.4)
    ax2.set_title("Seg length", fontsize=13, weight="bold", color="#e0e0e0")
    ax2.set_xlabel("Time (s)", fontsize=11, color="#8a95a5")
    ax2.set_ylabel("Seg length (px)", fontsize=11, color="#8a95a5")
    ax2.grid(True, color="#252e3b", linestyle="-", linewidth=0.5)
    ax2.set_xlim(t_slice.min(), t_slice.max())

    plt.tight_layout()
    plt.show()

plot_FR_and_kinetics(offset_sec=-1.367)

# Estimate time offset :: Cross correlation