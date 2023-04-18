import argparse
import asyncio
import logging
import math
import multiprocessing
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import aiohttp
import aiohttp_retry
import cv2


def _get_cache_videos(cache_file_dir: Path) -> list[str]:
    cache_videos = []
    try:
        with open(cache_file_dir, "r") as f:
            cache_videos.extend(f.read().splitlines())
        logger.info(f"Cache file {cache_file_dir} found. {len(cache_videos)} videos already processed.")
    except FileNotFoundError:
        logger.info(f"Cache file {cache_file_dir} not found. Creating a new one.")
        cache_file_dir.parent.mkdir(parents=True, exist_ok=True)
        open(cache_file_dir, "a").close()
    return cache_videos


def _set_client_session(**kwargs) -> aiohttp_retry.RetryClient:
    retry_options = aiohttp_retry.ExponentialRetry(
        **kwargs,
        exceptions={asyncio.exceptions.TimeoutError},
    )
    retry_client_session = aiohttp_retry.RetryClient(
        retry_options=retry_options,
        raise_for_status=True,
        logger=logger,
    )
    return retry_client_session


async def fetch(session: aiohttp.ClientSession, url: str) -> None:
    async with session.get(url) as response:
        if response.status != 200:
            response.raise_for_status()
        return await response.text()


def get_urls(response: str) -> list[str]:
    urls = re.findall(r'href=[\'"]?([^\'" >]+)', response)
    return urls


async def get_videos_urls(url: str, output_dir: str, cache_videos: list[str], **kwargs) -> None:
    videos = []
    retry_client_session = _set_client_session(**kwargs)
    async with retry_client_session as session:
        response = await fetch(session=session, url=url)
        years = [*filter(lambda x: re.match(r"^\d{4}/$", x), get_urls(response=response))]
        for year in years:
            url_year = url + year
            response = await fetch(session=session, url=url_year)
            months = [*filter(lambda x: re.match(r"^\d{2}/$", x), get_urls(response=response))]
            for month in months:
                url_month = url_year + month
                response = await fetch(session=session, url=url_month)
                days = [*filter(lambda x: re.match(r"^\d{2}/$", x), get_urls(response=response))]
                for day in days[:1]:
                    url_day = url_month + day
                    response = await fetch(session=session, url=url_day)

                    frame_path = Path(output_dir, year, month, day)
                    frame_path.mkdir(parents=True, exist_ok=True)
                    urls_day = [*map(lambda x: url_day + x, get_urls(response=response)[1:])]
                    urls_day = [*filter(lambda x: x.endswith(".mp4") and not x.endswith("quick.mp4"), urls_day)]
                    frame_file_names = [*map(lambda x: x.rsplit("/", 1)[-1].removesuffix(".mp4"), urls_day)]
                    frames_paths = [*map(lambda x: Path(frame_path, x), frame_file_names)]
                    if len(urls_day) > 0:
                        logger.info(f"Found {len(urls_day)} videos in {url_day}")
                        for url_day, frame_path in zip(urls_day, frames_paths):
                            if url_day in cache_videos:
                                logger.info(f"Video {url_day} already processed. Skipping.")
                                continue
                            videos.append((url_day, frame_path))
    return videos


def preprocess_video_frames(video: tuple[str, Path], frequency: float, cache_file_dir: Path) -> None:
    url, frames_path = video
    cap = cv2.VideoCapture(url)

    frames_path = str(frames_path)
    frames_count = 0
    # # Workaround for videos that stuck in an infinite loop
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames = math.ceil(fps * frequency)
    frames_expected = math.ceil(video_length / skip_frames)
    frames_obtained = 0

    while cap.isOpened() and frames_obtained < frames_expected:
        status, frame = cap.read()
        if not status:
            break

        file_path = f"{frames_path}_{frames_count}.png"
        frames_count += skip_frames
        cv2.imwrite(file_path, frame)
        frames_obtained += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frames_count)
    cap.release()

    with open(cache_file_dir, "a") as f:
        f.write(f"{url}\n")
    logger.info(f"Processed {url} at {fps} fps.")


async def main(url: str, output_dir: str, cache_file: str, frequency: float, num_processes: int, **kwargs) -> None:
    cache_file_dir = Path(output_dir, cache_file)
    cache_videos = _get_cache_videos(cache_file_dir=cache_file_dir)

    # FIXME: Change to use a producer(one)-consumer(multiple, one per process) pattern
    # to avoid having to get all videos urls before starting the processing.
    videos = await get_videos_urls(
        url=url,
        output_dir=output_dir,
        cache_videos=cache_videos,
        **kwargs,
    )
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        executor.map(
            partial(
                preprocess_video_frames,
                frequency=frequency,
                cache_file_dir=cache_file_dir,
            ),
            videos,
        )


if __name__ == "__main__":
    URL = "http://smartbay.marine.ie/data/video/aja-helo-1H000314/"

    logging.basicConfig(
        format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d:%H:%M:%S',
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description=f"Preprocess videos from SmartBay: {URL}")
    parser.add_argument("-o", "--Output", type=str, help="Output directory", default="frames")
    parser.add_argument("-c", "--Cache", type=str, help="Cache file", default=".cache")
    parser.add_argument("-f", "--Frequency", type=float, help="Frequency in seconds (save a frame every n seconds)", default=60.0)
    parser.add_argument("-p", "--Processes", type=int, help="Number of processes", default=multiprocessing.cpu_count())
    parser.add_argument("-a", "--Attempts", type=int, help="Number of retry attempts", default=5)
    parser.add_argument("-st", "--StartTimeout", type=float, help="Retry start timeout", default=0.5)

    args = parser.parse_args()

    asyncio.run(
        main(
            url=URL,
            output_dir=args.Output,
            cache_file=args.Cache,
            frequency=args.Frequency,
            num_processes=args.Processes,
            attempts=args.Attempts,
            start_timeout=args.StartTimeout,
        ),
    )
