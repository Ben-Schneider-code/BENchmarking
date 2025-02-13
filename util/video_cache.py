import os
from pathlib import Path

def init_cache():
    cache_location = os.path.join(Path.home(), ".cache/video-banchmark")
    if "CACHE" in os.environ:
        cache_location = os.path.join(os.environ["CACHE"], "video-banchmark")
    
    cache_location = Path(cache_location)
    cache_location.mkdir(parents=True, exist_ok=True)
