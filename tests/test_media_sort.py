from pathlib import Path

from tools.compose import _scene_media_sort_key


def test_scene_media_sort_key_orders_numerically():
    files = [Path('scene10.mp3'), Path('scene2.mp3'), Path('scene1.mp3')]
    ordered = sorted(files, key=_scene_media_sort_key)
    assert [path.name for path in ordered] == ['scene1.mp3', 'scene2.mp3', 'scene10.mp3']
