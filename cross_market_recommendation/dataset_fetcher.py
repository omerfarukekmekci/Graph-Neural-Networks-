import os
import requests
from bs4 import BeautifulSoup

base_url = "https://xmrec.github.io/data/us/"
r = requests.get(base_url)
soup = BeautifulSoup(r.text, "html.parser")

for link in soup.find_all("a"):
    href = link.get("href")
    if href.endswith(".gz"):
        url = base_url + href
        print("Downloading:", url)
        file = requests.get(url)
        with open(href, "wb") as f:
            f.write(file.content)