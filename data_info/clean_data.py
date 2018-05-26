import json


def clean_num(str):
    for i in range(9):
        str = str.replace('{}. '.format(i), '').replace('{}.'.format(i), '')
    return str


if __name__ == "__main__":
    with open('./annotation.json', 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        data[i]['paragraph'] = (data[i]['impression'] + ' ' + data[i]['findings']).strip()
        data[i]['paragraph'] = data[i]['paragraph'].lower().replace('/', '').replace(',', '')
        data[i]['paragraph'] = clean_num(data[i]['paragraph'])
    with open('./clean_annotation.json', 'w') as f:
        json.dump(data, f)
