import cv2
import numpy as np
import os
import subprocess
import argparse
import sys
import traceback  # Import traceback for detailed error reporting

def calculate_motion_score(frame1_gray, frame2_gray):
    """Calculates a motion score based on the Mean Absolute Difference."""
    diff = cv2.absdiff(frame1_gray, frame2_gray)
    score = np.mean(diff)
    return score

def clean_video(input_path, output_path, activity_threshold, min_segment_duration_sec, smoothing_window_sec, buffer_sec=0.0):
    """
    Analyzes a video for motion and removes inactive segments using FFmpeg.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the cleaned output video file.
        activity_threshold (float): Threshold for motion score. Frames with smoothed
                                     scores >= threshold are considered active.
        min_segment_duration_sec (float): Minimum duration (in seconds) for an
                                          active segment to be kept.
        smoothing_window_sec (float): Duration (in seconds) over which to average
                                      motion scores for smoothing.
        buffer_sec (float, optional):  Time (in seconds) to add before and after each active segment.
                                          Default is 0.0 (no buffer).
    """
    print(f"Starting video cleaning process for: {input_path}")
    print(f"Parameters: Threshold={activity_threshold}, Min Duration={min_segment_duration_sec}s, Smoothing={smoothing_window_sec}s, Buffer={buffer_sec}s")

    # --- 1. Video Analysis (OpenCV) ---
    print("Phase 1: Analyzing video for motion...")
    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {input_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0 or frame_count <= 0:
            cap.release()
            raise ValueError(f"Invalid video properties (FPS: {fps}, Frames: {frame_count}).")

        print(f"Video Info: {width}x{height}, {fps:.2f} FPS, {frame_count} frames (Duration: {frame_count / fps:.2f}s)")

        motion_scores = []
        prev_frame_gray = None
        frame_num = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            current_frame_gray = cv2.GaussianBlur(current_frame_gray, (21, 21), 0)

            if prev_frame_gray is not None:
                score = calculate_motion_score(prev_frame_gray, current_frame_gray)
                motion_scores.append(score)
            else:
                motion_scores.append(0)  # Score for the first frame

            prev_frame_gray = current_frame_gray
            frame_num += 1

            if frame_num % int(fps * 10) == 0:
                progress = (frame_num / frame_count) * 100
                print(f"  Analyzed frame {frame_num}/{frame_count} ({progress:.1f}%)", end='\r')

        cap.release()
        print(f"\nAnalysis complete. Calculated {len(motion_scores)} motion scores.")

        if not motion_scores:
            raise ValueError("No motion scores calculated (video might be empty or corrupted).")

    except Exception as e:
        print(f"Error during OpenCV analysis phase: {e}")
        if 'cap' in locals() and cap.isOpened():
            cap.release()
        return  # Important: Exit the function if analysis fails

    # --- 2. Temporal Smoothing & Segment Identification ---
    print("Phase 2: Identifying active segments...")
    smoothed_scores = None  # Initialize here
    try:
        smoothing_window_frames = max(1, int(smoothing_window_sec * fps))

        if len(motion_scores) >= smoothing_window_frames and smoothing_window_frames > 1:
            print(f"  Applying smoothing with window size: {smoothing_window_frames} frames")
            motion_scores_np = np.array(motion_scores, dtype=float)
            kernel = np.ones(smoothing_window_frames) / smoothing_window_frames
            smoothed_scores_conv = np.convolve(motion_scores_np, kernel, mode='valid')

            if len(smoothed_scores_conv) > 0:
                pad_start_len = smoothing_window_frames // 2
                pad_end_len = len(motion_scores) - len(smoothed_scores_conv) - pad_start_len
                padding_start = np.full(pad_start_len, smoothed_scores_conv[0])
                padding_end = np.full(pad_end_len, smoothed_scores_conv[-1])
                smoothed_scores = np.concatenate((padding_start, smoothed_scores_conv, padding_end))
                print(f"  Smoothing applied. Smoothed score length: {len(smoothed_scores)}")
            else:
                print("  Warning: Smoothing resulted in empty array. Using original motion scores.")
                smoothed_scores = np.array(motion_scores, dtype=float)
        else:
            print("  Skipping smoothing (video too short or smoothing window too small).")
            smoothed_scores = np.array(motion_scores, dtype=float)

        if smoothed_scores is None:
            raise ValueError("Smoothed scores were not calculated.")

        if len(smoothed_scores) != len(motion_scores):
            raise ValueError(
                f"Logic error: Smoothed scores length ({len(smoothed_scores)}) does not match original motion scores length ({len(motion_scores)})."
            )

        print("  Identifying active frames based on threshold...")
        active_frames = smoothed_scores >= activity_threshold
        raw_segments_to_keep = []
        in_active_segment = False
        start_frame = 0
        min_segment_frames = int(min_segment_duration_sec * fps)

        for i, is_active in enumerate(active_frames):
            if is_active and not in_active_segment:
                start_frame = i
                in_active_segment = True
            elif not is_active and in_active_segment:
                end_frame = i
                if (end_frame - start_frame) >= min_segment_frames:
                    raw_segments_to_keep.append((start_frame, end_frame))
                in_active_segment = False

        if in_active_segment:
            end_frame = len(active_frames)
            if (end_frame - start_frame) >= min_segment_frames:
                raw_segments_to_keep.append((start_frame, end_frame))

        if not raw_segments_to_keep:
            print("No active segments found meeting the criteria. No output file will be created.")
            return

        print(f"Identified {len(raw_segments_to_keep)} raw active segments.")

        # --- Apply Buffer and Merge ---
        if buffer_sec > 0 and raw_segments_to_keep:
            print(f"  Applying {buffer_sec:.2f}s buffer and merging segments...")
            buffer_frames = int(buffer_sec * fps)
            final_segments = []

            raw_segments_to_keep.sort(key=lambda x: x[0])  # Sort for merging

            current_start, current_end = raw_segments_to_keep[0]
            current_start = max(0, current_start - buffer_frames)
            current_end = min(len(active_frames), current_end + buffer_frames)

            for i in range(1, len(raw_segments_to_keep)):
                next_start, next_end = raw_segments_to_keep[i]
                adj_next_start = max(0, next_start - buffer_frames)
                adj_next_end = min(len(active_frames), next_end + buffer_frames)

                if adj_next_start <= current_end:
                    current_end = max(current_end, adj_next_end)
                else:
                    final_segments.append((current_start, current_end))
                    current_start = adj_next_start
                    current_end = adj_next_end

            final_segments.append((current_start, current_end))
            segments_to_keep = final_segments
            print(f"  Applied buffer. Resulted in {len(segments_to_keep)} final segments after merging.")
        else:
            segments_to_keep = raw_segments_to_keep

        if segments_to_keep:
            print("Final segments to keep (Frame numbers):")
            for start_f, end_f in segments_to_keep:
                print(f"  - Keep segment: {start_f} to {end_f} ({start_f / fps:.2f}s to {end_f / fps:.2f}s)")
        else:
            print("No final segments to keep after processing.")
            return

    except Exception as e:
        print(f"Error during segment identification phase: {e}")
        traceback.print_exc()
        return

    # --- 3. Video Editing (FFmpeg via subprocess) ---
    print("Phase 3: Extracting and concatenating segments using FFmpeg...")
    temp_files = []
    file_list_path = "mylist.txt"
    ffmpeg_path = "ffmpeg"

    try:
        try:
            subprocess.run([ffmpeg_path, "-version"], capture_output=True, check=True, text=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as ffmpeg_err:
            print(f"Error: FFmpeg command not found or failed. Ensure FFmpeg is installed and in your system's PATH.")
            print(f"Details: {ffmpeg_err}")
            return

        with open(file_list_path, "w") as f:
            for i, (start_f, end_f) in enumerate(segments_to_keep):
                start_time = start_f / fps
                duration = (end_f - start_f) / fps
                temp_output_path = f"temp_segment_{i}_{os.getpid()}.mp4"
                temp_files.append(temp_output_path)

                ffmpeg_extract_cmd_copy = [
                    ffmpeg_path,
                    '-ss', str(start_time),
                    '-i', input_path,
                    '-t', str(duration),
                    '-c', 'copy',
                    '-avoid_negative_ts', 'make_zero',
                    '-y',
                    temp_output_path
                ]
                print(f"  Extracting segment {i + 1}/{len(segments_to_keep)} ({start_time:.2f}s, duration {duration:.2f}s) [Attempting copy]...")
                result_copy = subprocess.run(ffmpeg_extract_cmd_copy, capture_output=True, text=True)

                if result_copy.returncode != 0:
                    print(f"    Warning: FFmpeg extraction (copy) failed for segment {i + 1}. Retrying with re-encoding.")
                    ffmpeg_extract_cmd_reencode = [
                        ffmpeg_path,
                        '-ss', str(start_time),
                        '-i', input_path,
                        '-t', str(duration),
                        '-y',
                        temp_output_path
                    ]
                    print(f"    Re-encoding segment {i + 1}...")
                    result_reencode = subprocess.run(ffmpeg_extract_cmd_reencode, capture_output=True, text=True)

                    if result_reencode.returncode != 0:
                        print(f"    Error: FFmpeg extraction failed even with re-encoding for segment {i + 1}.")
                        print(f"    FFmpeg stderr (re-encode): {result_reencode.stderr}")
                        if os.path.exists(temp_output_path):
                            try:
                                os.remove(temp_output_path)
                            except OSError:
                                pass
                        temp_files.remove(temp_output_path)
                        continue

                f.write(f"file '{os.path.basename(temp_output_path)}'\n")

        if not temp_files or not os.path.exists(file_list_path) or os.path.getsize(file_list_path) == 0:
            raise RuntimeError("No segments were successfully extracted. Cannot create output file.")

        ffmpeg_concat_cmd_copy = [
            ffmpeg_path,
            '-f', 'concat',
            '-safe', '0',
            '-i', file_list_path,
            '-c', 'copy',
            '-y',
            output_path
        ]
        print("Concatenating segments [Attempting copy]...")
        result_concat_copy = subprocess.run(ffmpeg_concat_cmd_copy, capture_output=True, text=True)

        if result_concat_copy.returncode != 0:
            print("  Warning: FFmpeg concatenation (copy) failed. Retrying with re-encoding.")
            ffmpeg_concat_cmd_reencode = [
                ffmpeg_path,
                '-f', 'concat',
                '-safe', '0',
                '-i', file_list_path,
                '-y',
                output_path
            ]
            print("  Re-encoding during concatenation...")
            result_concat_reencode = subprocess.run(ffmpeg_concat_cmd_reencode, capture_output=True, text=True)

            if result_concat_reencode.returncode != 0:
                raise RuntimeError(f"FFmpeg concatenation failed even with re-encoding.  FFmpeg stderr (re-encode): {result_concat_reencode.stderr}")

        print(f"Successfully created cleaned video: {output_path}")

    except Exception as e:
        print(f"An error occurred during FFmpeg processing phase: {e}")
        traceback.print_exc()  # Print the full traceback
    finally:
        print("Cleaning up temporary files...")
        if os.path.exists(file_list_path):
            try:
                os.remove(file_list_path)
            except OSError as e:
                print(f"  Warning: Could not remove list file {file_list_path}: {e}")
        temp_files_to_remove = list(temp_files)
        for temp_file in temp_files_to_remove:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError as e:
                    print(f"  Warning: Could not remove temp file {temp_file}: {e}")
        print("Cleanup complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean video by removing inactive segments.")
    parser.add_argument("input_video", help="Path to the input video file.")
    parser.add_argument("output_video", help="Path for the cleaned output video file.")
    parser.add_argument("-t", "--threshold", type=float, default=5.0,
                        help="Motion sensitivity threshold (lower=more sensitive). Default: 5.0")
    parser.add_argument("-d", "--duration", type=float, default=1.0,
                        help="Minimum duration (seconds) for active segments to be kept. Default: 1.0")
    parser.add_argument("-s", "--smoothing", type=float, default=0.5,
                        help="Smoothing window duration (seconds). Default: 0.5")
    parser.add_argument("-b", "--buffer", type=float, default=0.0,
                        help="Buffer time (seconds) to add around active segments. Default: 0.0")

    args = parser.parse_args()

    input_video = args.input_video
    output_video = args.output_video
    threshold = args.threshold
    min_duration = args.duration
    smoothing_window = args.smoothing
    buffer_time = args.buffer  # Get buffer time from arguments

    if not os.path.exists(input_video):
        print(f"Error: Input video file not found: {input_video}")
        sys.exit(1)

    output_dir = os.path.dirname(output_video)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    clean_video(input_video, output_video, threshold, min_duration, smoothing_window, buffer_time) # Pass buffer_time
