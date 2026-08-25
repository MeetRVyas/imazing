import cv2
import numpy as np
import pytest

from imazing import Imazing, VideoStream


@pytest.fixture
def sample_video_path(tmp_path):
    """A tiny real video file, generated on the fly -- no binary fixtures
    checked into the repo, no network access."""
    path = tmp_path / "sample.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (64, 48))
    rng = np.random.default_rng(seed=3)
    for _ in range(5):
        frame = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
        writer.write(frame)
    writer.release()
    return str(path)


class TestVideoStreamReading:
    def test_opens_a_real_video_file(self, sample_video_path):
        vs = VideoStream(sample_video_path)
        assert vs.width == 64
        assert vs.height == 48
        vs.release()

    def test_read_frame_returns_imazing_instances(self, sample_video_path):
        vs = VideoStream(sample_video_path)
        frame = vs.read_frame()
        assert isinstance(frame, Imazing)
        assert frame.image.shape == (48, 64, 3)
        vs.release()

    def test_reads_all_frames_then_returns_none(self, sample_video_path):
        vs = VideoStream(sample_video_path)
        count = 0
        while vs.read_frame() is not None:
            count += 1
        assert count == 5
        vs.release()


class TestVideoStreamBadSource:
    def test_nonexistent_source_does_not_raise_on_construction(self):
        vs = VideoStream("this_video_does_not_exist_12345.mp4")
        assert vs.cap is not None
        vs.release()

    def test_read_frame_on_bad_source_returns_none_not_an_exception(self):
        vs = VideoStream("this_video_does_not_exist_12345.mp4")
        assert vs.read_frame() is None
        vs.release()


class TestVideoStreamRecording:
    def test_start_recording_write_and_release_roundtrip(self, sample_video_path, tmp_path):
        vs = VideoStream(sample_video_path)
        out_path = tmp_path / "out.mp4"
        vs.start_recording(str(out_path), fps=10.0)
        frame = vs.read_frame()
        vs.write_frame(frame)
        vs.release()
        assert out_path.exists()
        assert out_path.stat().st_size > 0
