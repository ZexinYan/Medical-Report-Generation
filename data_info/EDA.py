import json

static_tags_name = ['nodule', 'borderline', 'spinal fusion', 'cardiac shadow', 'interstitial', 'pulmonary congestion',
                    'technical quality of image unsatisfactory', 'bronchiectasis', 'cervical vertebrae',
                    'hypoinflation', 'medical device', 'prominent', 'mass', 'breast implants', 'calcinosis',
                    'aortic aneurysm', 'aorta, thoracic', 'lower lobe', 'scattered', 'left', 'lung, hyperlucent',
                    'pneumoperitoneum', 'enlarged', 'foreign bodies', 'epicardial fat', 'reticular', 'abnormal',
                    'irregular', 'obscured', 'large', 'diaphragm', 'right', 'breast', 'pulmonary edema',
                    'hyperostosis, diffuse idiopathic skeletal', 'airspace disease', 'stents', 'mild', 'volume loss',
                    'shift', 'sulcus', 'humerus', 'lucency', 'blunted', 'osteophyte', 'blood vessels',
                    'lumbar vertebrae', 'flattened', 'tortuous', 'small', 'healed', 'hypertension, pulmonary',
                    'bone diseases, metabolic', 'trachea', 'atherosclerosis', 'mediastinum', 'coronary vessels', 'lung',
                    'chronic', 'multiple', 'ribs', 'pulmonary disease, chronic obstructive', 'apex', 'hilum',
                    'spondylosis', 'diffuse', 'paratracheal', 'pneumothorax', 'clavicle', 'retrocardiac', 'lymph nodes',
                    'bronchovascular', 'azygos lobe', 'pulmonary emphysema', 'granulomatous disease',
                    'calcified granuloma', 'normal', 'thoracic vertebrae', 'funnel chest', 'thorax', 'aorta',
                    'adipose tissue', 'anterior', 'arthritis', 'emphysema', 'fractures, bone', 'hernia, hiatal',
                    'implanted medical device', 'sutures', 'granuloma', 'pleura', 'thickening', 'cysts', 'upper lobe',
                    'middle lobe', 'pleural effusion', 'deformity', 'contrast media', 'pulmonary atelectasis',
                    'hyperdistention', 'pericardial effusion', 'spine', 'mastectomy', 'surgical instruments',
                    'nipple shadow', 'heart', 'streaky', 'blister', 'catheters, indwelling', 'bilateral', 'neck',
                    'cavitation', 'density', 'scoliosis', 'pulmonary artery', 'round', 'opacity',
                    'lung diseases, interstitial', 'sternum', 'heart ventricles', 'lingula', 'aortic valve',
                    'heart failure', 'heart atria', 'sarcoidosis', 'bullous emphysema', 'sclerosis',
                    'costophrenic angle', 'kyphosis', 'hydropneumothorax', 'consolidation', 'dislocations', 'markings',
                    'abdomen', 'tube, inserted', 'no indexing', 'pneumonectomy', 'posterior', 'patchy',
                    'diaphragmatic eventration', 'pulmonary fibrosis', 'pneumonia', 'cardiomegaly', 'focal', 'cicatrix',
                    'elevated', 'infiltrate', 'moderate', 'degenerative', 'base', 'trachea, carina', 'severe',
                    'bronchi', 'pulmonary alveoli', 'shoulder', 'cystic fibrosis']


def count_tags_num(data, k):
    tags = dict()
    for each in data:
        for tag in data[each]:
            if tag in tags:
                tags[tag] += 1
            else:
                tags[tag] = 1
    clean_tag = dict()
    total = 0
    for each in tags:
        if tags[each] > k:
            clean_tag[each] = tags[each]
            total += clean_tag[each]
    print(clean_tag)
    print(total)
    return clean_tag


def count_total_num(data):
    total = 0
    for each in data:
        total += len(data[each])
    print("Total tag num:{}".format(total))
    print("Average tag num:{}".format(total / len(data)))


if __name__ == '__main__':
    test_data = None
    val_data = None
    train_data = None
    # data_annotation = None

    # with open('new_annotation.json', 'r') as f:
    #     data_annotation = json.load(f)

    # with open('test_data.json', 'r') as f:
    #     test_data = json.load(f)
    #
    # with open('val_data.json', 'r') as f:
    #     val_data = json.load(f)
    #
    # with open('train_data.json', 'r') as f:
    #     train_data = json.load(f)

    # print(len(count_tags_num(data_annotation, 70)))

    #
    # train_annotation = []
    # val_annotation = []
    # test_annotation = []
    #
    # for each in data_annotation:
    #     if each['image_id'][:-4] in list(train_data.keys()):
    #         each['tags'] = train_data[each['image_id'][:-4]]
    #         train_annotation.append(each)
    #     if each['image_id'][:-4] in list(val_data.keys()):
    #         each['tags'] = val_data[each['image_id'][:-4]]
    #         val_annotation.append(each)
    #     if each['image_id'][:-4] in list(test_data.keys()):
    #         each['tags'] = test_data[each['image_id'][:-4]]
    #         test_annotation.append(each)
    #
    # with open('train_annotation.json', 'w') as f:
    #     json.dump(train_annotation, f)
    #
    # with open('val_annotation.json', 'w') as f:
    #     json.dump(val_annotation, f)
    #
    # with open('test_annotation.json', 'w') as f:
    #     json.dump(test_annotation, f)

    # count_total_num(train_data)
    #
    # count_total_num(val_data)
    #
    # count_total_num(test_data)

    with open('annotation.json', 'r') as f:
        data = json.load(f)
    total_sentence = 0
    total_word = 0

    for each in data:
        total_sentence += len(each['findings'].split('. '))
        for sentence in each['findings'].split('. '):
            total_word += len(sentence.split())
    for each in data:
        total_sentence += len(each['impression'].split('. '))
        for sentence in each['impression'].split('. '):
            total_word += len(sentence.split())

    print(len(data))
    print(total_sentence)
    print(total_word)
    print("Average num of sentences:{}".format(total_sentence / len(data)))
    print('Average num of words for each sentence:{}'.format(total_word / total_sentence))
