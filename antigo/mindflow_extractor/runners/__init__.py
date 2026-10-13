"""Runners: pontos de entrada para captura e processamento."""
from .webcam_runner import run_webcam
from .video_runner import process_video

__all__ = ["run_webcam", "process_video"]
