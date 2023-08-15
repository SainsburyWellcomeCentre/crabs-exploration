from pathlib import Path

from bboxes_labelling.extract_frames_to_label_w_sleap import get_list_of_sleap_videos

sample_input_dir = Path(__file__).parent / "data"


def test_extension_case_insensitive():
    list_video_locations = [sample_input_dir]
    list_video_extensions = ["MP4"]

    list_sleap_videos = get_list_of_sleap_videos(
        list_video_locations, list_video_extensions
    )
    assert len(list_sleap_videos) == 2
