import json
from bs4 import BeautifulSoup
import requests
import time


def get_url(image_name):
    base_url = 'https://openi.nlm.nih.gov/detailedresult.php?img={}' \
               '&query=&coll=cxr&req=4&simResults=CXR100_IM-0002-1001&npos=95'.format(image_name)
    return base_url


def get_content(url):
    session = requests.session()
    session.headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5)'
                                     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    r = session.get(url)
    return r.content


def get_tags(content):
    soup = BeautifulSoup(content)
    tags = set()
    for child in soup.find(attrs={'class': 'mli'}):
        for tag in child.text.split('/'):
            tags.add(tag.strip())
    return list(tags)


with open('annotation.json', 'r') as f:
    data = json.load(f)
    with open('./new_annotation_1.json', 'r') as json_file:
        new_data = json.load(json_file)
    for item in data:
        try:
            image_id = item['image_id'][:-4]
            if image_id not in new_data:
                url = get_url(image_id)
                content = get_content(url)
                tags = get_tags(content)
                new_data[image_id] = tags
                time.sleep(1)
                print('id: {}; tags: {}'.format(image_id, tags))
        except Exception as err:
            print(err)
            time.sleep(5)
        with open("./new_annotation.json", 'w') as json_file_1:
            json.dump(new_data, json_file_1, ensure_ascii=False)
