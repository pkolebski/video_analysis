from tqdm import tqdm
import urllib.request
import os


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    if not os.path.isfile(output_path):
        if not os.path.exists(output_path.split('/')[0:-1][0]):
            os.makedirs(output_path.split('/')[0:-1][0])
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
    else:
        print("File ", output_path, " exist")
