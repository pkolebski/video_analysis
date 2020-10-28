import os
from utils.download import download_url

DOWNLOADS_DIR = 'data'


def main():
    for url in open('datasets_urls.txt'):
        url = url[0:-1]
        name = url.rsplit('/', 1)

        filename = os.path.join(DOWNLOADS_DIR, name[1])


            # print("Downloading", filename)
        download_url(url, filename)


if __name__ == "__main__":
    main()

