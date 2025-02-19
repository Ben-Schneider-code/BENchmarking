from pytubefix import YouTube
from pytubefix.cli import on_progress
import sys

url = sys.argv[1]

yt = YouTube(url, use_oauth=True, allow_oauth_cache=True, on_progress_callback = on_progress)
print(yt.title)

ys = yt.streams.get_highest_resolution()
ys.download()