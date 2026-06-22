"""
Video Encoder: FFmpeg H.264 Encoding Pipeline.

Handles the technical aspects of video encoding:
- FFmpeg subprocess management
- Raw BGR24 frame piping
- H.264/MP4 output

Separated from rendering for single responsibility.
"""

import subprocess
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class EncoderConfig:
    """Configuration for video encoding."""
    fps: int = 20
    crf: int = 28  # Quality: 18=high, 28=medium, lower=better
    preset: str = "ultrafast"
    pixel_format: str = "yuv420p"


class VideoEncoder:
    """
    High-performance video encoder using FFmpeg.

    Streams raw BGR24 frames to FFmpeg for real-time H.264 encoding.
    No disk I/O during encoding - direct pipe to FFmpeg.

    Usage:
        encoder = VideoEncoder("output.mp4", width=600, height=600)
        for frame in frames:
            encoder.add_frame(frame)
        encoder.close()

    Or with context manager:
        with VideoEncoder("output.mp4", 600, 600) as encoder:
            for frame in frames:
                encoder.add_frame(frame)
    """

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        config: Optional[EncoderConfig] = None,
    ):
        """
        Initialize the video encoder.

        Args:
            output_path: Path for output MP4 file
            width: Frame width in pixels
            height: Frame height in pixels
            config: Encoder configuration (uses defaults if None)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.config = config or EncoderConfig()
        self.process = None
        self.frame_count = 0

        if output_path:
            self._start_ffmpeg()

    def _start_ffmpeg(self) -> None:
        """Start FFmpeg subprocess for encoding."""
        cfg = self.config

        cmd = [
            "ffmpeg",
            "-y",                                  # Overwrite output
            "-f", "rawvideo",                      # Input format
            "-vcodec", "rawvideo",                 # Input codec
            "-s", f"{self.width}x{self.height}",   # Frame size
            # Input pixel format (OpenCV)
            "-pix_fmt", "bgr24",
            "-r", str(cfg.fps),                    # Frame rate
            "-i", "-",                             # Read from stdin
            "-c:v", "libx264",                     # Output codec
            "-preset", cfg.preset,                 # Encoding speed
            "-crf", str(cfg.crf),                  # Quality
            "-pix_fmt", cfg.pixel_format,          # Output pixel format
            self.output_path
        ]

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def add_frame(self, frame: np.ndarray) -> None:
        """
        Write a frame to the video.

        Args:
            frame: BGR image (height, width, 3) as uint8
        """
        if self.process and self.process.stdin:
            self.process.stdin.write(frame.tobytes())
            self.frame_count += 1

    def close(self) -> None:
        """
        Finalize video and close FFmpeg.

        MUST be called after all frames are added.
        """
        if self.process:
            # Close stdin to signal end of input to ffmpeg
            if self.process.stdin:
                self.process.stdin.close()

            # Drain stderr to avoid deadlock, then wait for process
            stderr_output = b""
            if self.process.stderr:
                stderr_output = self.process.stderr.read()
                self.process.stderr.close()

            # Now wait for the process to finish
            return_code = self.process.wait()

            if return_code != 0:
                error_msg = (
                    stderr_output.decode()[-500:]
                    if stderr_output
                    else "unknown error"
                )
                print(f"⚠️ FFmpeg warning (code {return_code}): {error_msg}")

            self.process = None

    def get_frame_count(self) -> int:
        """Return number of frames written."""
        return self.frame_count

    def get_duration(self) -> float:
        """Return estimated video duration in seconds."""
        return self.frame_count / self.config.fps

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
