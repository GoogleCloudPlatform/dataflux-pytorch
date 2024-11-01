import os
import time
import urllib.request
from multiprocessing.dummy import Pool as ThreadPool

URLS_FILE_NAME = "urls.txt"
URL_PREFIX = "https://data.together.xyz/redpajama-data-1T/v1.0.0/"
NUM_THREADS = 32


def download_to_disk(url):
    dest_filepath = url.removeprefix(URL_PREFIX)
    # set request header
    opener = urllib.request.URLopener()
    opener.addheader('User-Agent', 'dummy')
    opener.retrieve(url, dest_filepath)

    print(f"Downloaded {url} to {dest_filepath}")


def get_urls_list(filename):
    with open(filename) as f:
        urls = [line.rstrip() for line in f]
    return urls


def create_parent_dirs(urls):
    for u in urls:
        filepath = u.removeprefix(URL_PREFIX)
        dirname = os.path.dirname(filepath)
        if not os.path.exists(dirname):
            print(f"Creating {dirname}...")
            os.makedirs(dirname, exist_ok=True)


if __name__ == "__main__":

    urls = get_urls_list(URLS_FILE_NAME)
    create_parent_dirs(urls)

    start = time.time()
    pool = ThreadPool(NUM_THREADS)
    results = pool.map(download_to_disk, urls)
    pool.close()
    pool.join()
    end = time.time()

    print(f"Downloaded {len(urls)} files in {end - start} seconds.")
