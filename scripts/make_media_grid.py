"""Create a 5x1 grid GIF from videos in the media folder."""
import argparse
from pathlib import Path

import cv2
import imageio
import numpy as np
from loguru import logger
from tqdm import tqdm


def load_video_frames(
    video_path: str, max_frames: int | None = None
) -> list[np.ndarray]:
    """Load frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

        if max_frames and len(frames) >= max_frames:
            break

    cap.release()
    return frames


def resize_frame(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize frame to target size."""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)


def create_grid_frame(
    video_frames: list[list[np.ndarray]],
    frame_idx: int,
    grid_shape: tuple[int, int],
    cell_size: tuple[int, int],
) -> np.ndarray:
    """Create a single grid frame from multiple videos at a given frame index."""
    rows, cols = grid_shape
    cell_w, cell_h = cell_size

    grid = np.zeros((rows * cell_h, cols * cell_w, 3), dtype=np.uint8)

    for i, frames in enumerate(video_frames):
        if i >= rows * cols:
            break

        row = i // cols
        col = i % cols

        # Get frame (loop if video is shorter)
        if len(frames) > 0:
            f_idx = frame_idx % len(frames)
            frame = resize_frame(frames[f_idx], cell_size)

            y_start = row * cell_h
            x_start = col * cell_w
            grid[y_start : y_start + cell_h, x_start : x_start + cell_w] = frame

    return grid


def main():
    parser = argparse.ArgumentParser(description="Create a grid GIF from videos")
    parser.add_argument(
        "--media-dir",
        type=str,
        default="media",
        help="Directory containing video files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="media/grid_visualization.gif",
        help="Output GIF path",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of rows in grid",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=5,
        help="Number of columns in grid",
    )
    parser.add_argument(
        "--cell-width",
        type=int,
        default=200,
        help="Width of each cell in pixels",
    )
    parser.add_argument(
        "--cell-height",
        type=int,
        default=200,
        help="Height of each cell in pixels",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output GIF frame rate",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to include (None for full length)",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of loops (0 for infinite)",
    )
    args = parser.parse_args()

    media_dir = Path(args.media_dir)

    # Find all video files
    video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    video_files = sorted(
        [f for f in media_dir.iterdir() if f.suffix.lower() in video_extensions]
    )

    if not video_files:
        logger.warning(f"No video files found in {media_dir}")
        return

    logger.info(f"Found {len(video_files)} videos")
    for v in video_files:
        logger.debug(f"  - {v.name}")

    # Limit to grid size
    max_videos = args.rows * args.cols
    if len(video_files) > max_videos:
        logger.info(f"Using first {max_videos} videos for {args.rows}x{args.cols} grid")
        video_files = video_files[:max_videos]

    # Load all video frames
    logger.info("Loading videos...")
    all_video_frames = []
    for video_path in tqdm(video_files, desc="Loading"):
        frames = load_video_frames(str(video_path), args.max_frames)
        all_video_frames.append(frames)
        logger.debug(f"  {video_path.name}: {len(frames)} frames")

    # Find the longest video
    max_frame_count = max(len(f) for f in all_video_frames)
    if args.max_frames:
        max_frame_count = min(max_frame_count, args.max_frames)

    logger.info(f"Creating {args.rows}x{args.cols} grid with {max_frame_count} frames")

    # Create grid frames
    cell_size = (args.cell_width, args.cell_height)
    grid_shape = (args.rows, args.cols)

    grid_frames = []
    for frame_idx in tqdm(range(max_frame_count), desc="Creating grid"):
        grid_frame = create_grid_frame(
            all_video_frames,
            frame_idx,
            grid_shape,
            cell_size,
        )
        grid_frames.append(grid_frame)

    # Save as GIF
    logger.info(f"Saving GIF to {args.output}")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    duration = 1000 / args.fps  # milliseconds per frame
    imageio.mimsave(
        str(output_path),
        grid_frames,
        format="GIF",
        duration=duration,
        loop=args.loop,
    )

    logger.success(
        f"Done! Grid size: {args.cols * args.cell_width}x{args.rows * args.cell_height}"
    )


if __name__ == "__main__":
    main()
