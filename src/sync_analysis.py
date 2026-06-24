"""
sync_analysis.py

Syncronize FR(Firing Rate) with limb kinetics
1. Calculate FR from spike2 spike wavemarks
2. Estimate time offset best fits
3. Find correlation coefficient
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.stats import gaussian_kde
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import cv2

def get_video_properties(video_path):
    # Get fps and total frames from video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None, None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    cap.release()
    
    return fps, total_frames

def normalize_array(arr, mode="min-max"):
    """
    Normalize an array using specified mode.
    Args:
        arr (np.array): Input array to normalize
        mode (str): Normalization mode, options are "z-score" or "min-max"
    Returns:
        np.array: Normalized array
    """
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
    # Calculate Firing Rate (FR) using Kernel Density Estimation (KDE)
    with open("./data/spike_times.txt", "r") as f:
        data = [float(line.strip()) for line in f if line.strip()]

    f_data = np.array(data)
    kde = gaussian_kde(f_data)
    kde.set_bandwidth(bw_method=bandwidth)
    y_density = kde(timescale) * len(f_data)

    return y_density

def calculate_kinetics(csv_path, mode = "seglen"):
    """
    Calculate kinetics from CSV file
    Args:
        csv_path (str): Path to exported CSV file from DeepLabCut
        mode (str): Mode for calculating kinetics, options are "seglen", "diff_seglen", "ankle_velocity", or "knee_velocity"
    Returns:
        np.array: Calculated kinetics
    """
    try:
        df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    except FileNotFoundError:
        print(f"Error: File not found - {csv_path}")
        return None

    # Extract knee and ankle header
    try:
        knee_x = df.xs(('knee', 'x'), level=(1, 2), axis=1).iloc[:, 0]
        knee_y = df.xs(('knee', 'y'), level=(1, 2), axis=1).iloc[:, 0]
        ankle_x = df.xs(('ankle', 'x'), level=(1, 2), axis=1).iloc[:, 0]
        ankle_y = df.xs(('ankle', 'y'), level=(1, 2), axis=1).iloc[:, 0]
    except KeyError as e:
        print(f"Column could not find Error: {e}")
        return None

    if mode == "seglen":
        # Seg length
        seg_length = np.sqrt((ankle_x - knee_x)**2 + (ankle_y - knee_y)**2)
        return seg_length
    
    elif mode == "diff_seglen":
        # Diff of seg length
        seg_length = np.sqrt((ankle_x - knee_x)**2 + (ankle_y - knee_y)**2)
        diff_seg_length = np.diff(seg_length, prepend=seg_length[0])
        return diff_seg_length
    
    elif mode == "ankle_velocity":
        # Ankle Velocity
        velocity = np.sqrt(np.diff(ankle_x, prepend=ankle_x[0])**2 + np.diff(ankle_y, prepend=ankle_y[0])**2)
        return velocity
    
    elif mode == "knee_velocity":
        # Knee Velocity
        velocity = np.sqrt(np.diff(knee_x, prepend=knee_x[0])**2 + np.diff(knee_y, prepend=knee_y[0])**2)
        return velocity

    else:
        raise ValueError("Invalid mode. Valid options are 'seglen', 'diff_seglen', 'ankle_velocity', or 'knee_velocity'.")

def find_best_offset(FR, kinetics, timescale, start_sec, end_sec, video_fps, max_lag=3.0, plot=False):
    """
    Brute-force search for best offset (with step size of 1/frame (s)) that maximizes Pearson r.

    Args:
        FR (np.array): Firing rate density array
        kinetics (np.array): Kinetics array
        timescale (np.array): Time scale array
        start_sec (float): Start time in seconds for the reference window
        end_sec (float): End time in seconds for the reference window
        video_fps (float): Video frames per second
        max_lag (float): Maximum lag in seconds to search for the best offset
        plot (bool): Whether to plot Pearson R vs. offset
    Returns:
        float: Best offset in seconds that maximizes Pearson r
    """
    step = 1.0 / video_fps

    # reference (fixed) window for FR
    mask_ref = (timescale >= start_sec) & (timescale <= end_sec)
    y_ref = normalize_array(FR[mask_ref])

    if len(y_ref) < 2 or np.std(y_ref) == 0:
        return 0.0

    best_offset = None
    best_r = -np.inf

    offsets = np.arange(-max_lag, max_lag + step / 2, step)
    r_list = []
    for off in offsets:
        mask_off = (timescale >= start_sec + off) & (timescale <= end_sec + off)
        kinetics_offset = kinetics[mask_off]

        if len(kinetics_offset) != len(y_ref):
            r_list.append(np.nan)
            continue

        kinetics_norm = normalize_array(kinetics_offset)
        r, _ = pearsonr(y_ref, kinetics_norm)
        r_list.append(r)

        if r > best_r:
            best_r = r
            best_offset = off
    
    if best_offset is None:
        print("No valid offset found in search range; using 0.0 s")
        return 0.0
    else:
        print(f"Best offset found: {best_offset:.3f} s")

    if plot:
        os.makedirs('./results', exist_ok=True)
        fig = plt.figure(figsize=(10, 5))
        plt.plot(offsets, r_list, marker='o', linestyle='-', color='b')
        plt.axvline(best_offset, color='r', linestyle='--', label=f'Best Offset: {best_offset:.3f} s')
        plt.title('Pearson R vs. Offset')
        plt.xlabel('Offset (s)')
        plt.ylabel('Pearson R')
        plt.legend()
        plt.grid()
        fig.savefig(os.path.join('./results', 'pearson_vs_offset.png'), dpi=200)
        plt.close(fig)
    
    return best_offset

def plot_FR_and_kinetics(t_slice, FR_slice, kinetics_slice, mode):
    """
    Plot Firing Rate and Kinetics in a two-panel figure.

    Args:
        t_slice (np.array): Time slice for plotting
        FR_slice (np.array): Firing Rate slice for plotting
        kinetics_slice (np.array): Kinetics slice for plotting
        mode (str): Mode for kinetics, used for labeling the plot
    """
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 8), sharex=True, facecolor="#11161d")

    ax1.plot(t_slice, FR_slice, color="#5897e5", linewidth=1.5, label="Frequency (KDE)")
    ax1.fill_between(t_slice, FR_slice, color="#20334e", alpha=0.5)
    ax1.set_title("Firing Rate", fontsize=13, weight="bold", color="#e0e0e0")
    ax1.set_ylabel("Frequency (#/s)", fontsize=11, color="#8a95a5")
    ax1.grid(True, color="#252e3b", linestyle="-", linewidth=0.5)
    ax1.set_xlim(t_slice.min(), t_slice.max())

    ax2.plot(t_slice, kinetics_slice, color="#34b360", linewidth=1.2)
    ax2.fill_between(t_slice, kinetics_slice, color="#153d23", alpha=0.4)
    if mode == "seglen":
        ax2.set_title("Segment Length", fontsize=13, weight="bold", color="#e0e0e0")
        ax2.set_ylabel("Seg length (px)", fontsize=11, color="#8a95a5")
    elif mode == "diff_seglen":
        ax2.set_title("Change in Segment Length", fontsize=13, weight="bold", color="#e0e0e0")
        ax2.set_ylabel("Δ Seg length (px)", fontsize=11, color="#8a95a5")
    elif mode == "ankle_velocity":
        ax2.set_title("Ankle Velocity", fontsize=13, weight="bold", color="#e0e0e0")
        ax2.set_ylabel("Ankle velocity (px/s)", fontsize=11, color="#8a95a5")
    elif mode == "knee_velocity":
        ax2.set_title("Knee Velocity", fontsize=13, weight="bold", color="#e0e0e0")
        ax2.set_ylabel("Knee velocity (px/s)", fontsize=11, color="#8a95a5")
    ax2.set_xlabel("Time (s)", fontsize=11, color="#8a95a5")
    ax2.grid(True, color="#252e3b", linestyle="-", linewidth=0.5)
    ax2.set_xlim(t_slice.min(), t_slice.max())

    plt.tight_layout()
    os.makedirs('./results', exist_ok=True)
    fig.savefig(os.path.join('./results', 'fr_and_kinetics.png'), dpi=200)
    plt.close(fig)


# Initialize video properties and time scale
sample_video_path = "./data/20250710_173046.mp4"
sample_csv_path = "./data/20250710_173046DLC_Resnet50_rabbit_analysisJun22shuffle1_snapshot_200.csv"
kinetics_mode_offset = "seglen"  # Options: "seglen", "diff_seglen", "ankle_velocity", "knee_velocity"
kinetics_mode = "seglen"

video_fps, total_frames = get_video_properties(video_path=sample_video_path)
timescale = np.arange(1/video_fps, total_frames/video_fps, 1/video_fps)
start_sec = 220
# end_sec = total_frames / video_fps  # Entire video duration
end_sec = 270


# Calculate Firing Rate and Kinetics, offset, and correlation stats
FR = calculate_FR()
offset_kinetics = calculate_kinetics(csv_path=sample_csv_path, mode=kinetics_mode_offset) # Calculate kinetics for offset estimation

offset = find_best_offset(FR, offset_kinetics, timescale, start_sec, end_sec, video_fps, max_lag=3.0, plot=True)

kinetics = calculate_kinetics(csv_path=sample_csv_path, mode=kinetics_mode) # Recalculate kinetics for correlation analysis

mask = (timescale >= start_sec) & (timescale <= end_sec)
mask_offset = (timescale >= start_sec + offset) & (timescale <= end_sec + offset)
t_slice = timescale[mask]
FR_slice = normalize_array(FR[mask])
kinetics_slice = normalize_array(kinetics[mask_offset])

pearson_r, p_val_p = pearsonr(FR_slice, kinetics_slice)
spearman_r, p_val_s = spearmanr(FR_slice, kinetics_slice)

print(f"Pearson R: {pearson_r:.4f} (p-value: {p_val_p})")
print(f"Spearman R: {spearman_r:.4f} (p-value: {p_val_s})")


# plot FR and kinetics
plot_FR_and_kinetics(t_slice, FR_slice, kinetics_slice, mode=kinetics_mode)