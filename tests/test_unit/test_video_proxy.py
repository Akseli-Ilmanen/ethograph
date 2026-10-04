"""The proxy cache names one file per (source, height)."""

from ethograph.io.video_proxy import DEFAULT_PROXY_HEIGHT, PROXY_HEIGHTS, proxy_cache_path


def test_each_height_has_its_own_cache_file(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    paths = {proxy_cache_path(video, tmp_path, height) for height in PROXY_HEIGHTS}
    assert len(paths) == len(PROXY_HEIGHTS), "two heights share a proxy file: one would be served as the other"


def test_default_height_keeps_the_name_of_proxies_cached_before_the_choice(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    assert proxy_cache_path(video, tmp_path, DEFAULT_PROXY_HEIGHT) == proxy_cache_path(video, tmp_path)
    assert "_" + str(DEFAULT_PROXY_HEIGHT) + "p" not in proxy_cache_path(video, tmp_path).name
