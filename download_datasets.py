import os
import urllib.request
from tqdm import tqdm

DOWNLOADS_DIR = 'data'


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def main():
    if not os.path.exists(DOWNLOADS_DIR):
        os.makedirs(DOWNLOADS_DIR)

    for url in open('datasets_urls.txt'):
        url = url[0:-1]
        name = url.rsplit('/', 1)

        filename = os.path.join(DOWNLOADS_DIR, name[1])

        if not os.path.isfile(filename):
            # print("Downloading", filename)
            download_url(url, filename)


if __name__ == "__main__":
    main()

