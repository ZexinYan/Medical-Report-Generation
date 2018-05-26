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

weights = [0.06054117907637786, 0.026579054228653698, 0.013289527114326849, 0.033592971316770644, 0.08047546974786814,
           0.04134519546679464, 0.05906456495256377, 0.0014766141238140942, 0.01550444830004799, 0.18088523016722655,
           0.013658680645280372, 0.07604562737642585, 0.008859684742884566, 0.0022149211857211415, 0.16907231717671378,
           0.0014766141238140942, 0.07752224150023995, 0.1081619845693824, 0.04688249843109749, 0.34331278378677693,
           0.01033629886669866, 0.0014766141238140942, 0.048359112554911585, 0.017719369485769132,
           0.0070139170881169475, 0.004060688840488759, 0.002584074716674665, 0.0066447635571634245,
           0.014766141238140943, 0.029532282476281885, 0.08342869799549632, 0.4197275646941563, 0.007383070619070471,
           0.02694820775960722, 0.0018457676547676178, 0.07235409206689061, 0.01033629886669866, 0.4260031747203662,
           0.00516814943334933, 0.004060688840488759, 0.011812912990512753, 0.010705452397652183, 0.015135294769094466,
           0.04429842371442283, 0.03949942781202702, 0.007752224150023995, 0.0313780501310495, 0.03027058953818893,
           0.1439698770718742, 0.09302668980028794, 0.02768651482151427, 0.0029532282476281884, 0.017350215954815607,
           0.0036915353095352357, 0.06866255675735539, 0.06201779320019196, 0.004429842371442283, 0.9343275868433681,
           0.04503673077632987, 0.15836686477906162, 0.05869541142161025, 0.028424821883421315, 0.0579571043597032,
           0.1159142087194064, 0.03802281368821293, 0.022518365388164936, 0.014766141238140943, 0.014766141238140943,
           0.01144375945955923, 0.015135294769094466, 0.07235409206689061, 0.05758795082874968, 0.0036915353095352357,
           0.024364133042932556, 0.06607848204068072, 0.1568902506552475, 0.853113810033593, 0.2960611318247259,
           0.0014766141238140942, 0.04429842371442283, 0.19011406844106463, 0.0022149211857211415, 0.03100889660009598,
           0.018826830078629703, 0.035438738971538264, 0.049835726678725684, 0.024733286573886078, 0.03027058953818893,
           0.0036915353095352357, 0.030639743069142456, 0.03506958544058474, 0.02768651482151427, 0.0022149211857211415,
           0.1033629886669866, 0.05426556905016797, 0.08084462327882166, 0.06201779320019196, 0.0007383070619070471,
           0.19085237550297168, 0.12920373583373326, 0.0033223817785817122, 0.12034405109084868, 0.0018457676547676178,
           0.05352726198826092, 0.004060688840488759, 0.013289527114326849, 0.04688249843109749, 0.0022149211857211415,
           0.06940086381926243, 0.2584074716674665, 0.0022149211857211415, 0.0014766141238140942, 0.06497102144782015,
           0.05278895492635387, 0.009597991804791612, 0.050943187271586254, 0.26246816050795524, 0.008121377680977518,
           0.0033223817785817122, 0.0022149211857211415, 0.05537302964302854, 0.0029532282476281884,
           0.0036915353095352357, 0.0018457676547676178, 0.0007383070619070471, 0.0014766141238140942,
           0.0036915353095352357, 0.05611133670493558, 0.018088523016722653, 0.0022149211857211415,
           0.016242755361955036, 0.0014766141238140942, 0.09930229982649784, 0.02916312894532836, 0.015135294769094466,
           0.05574218317398206, 0.002584074716674665, 0.04060688840488759, 0.05426556905016797, 0.021041751264350844,
           0.011074605928605707, 0.026579054228653698, 0.19675883199822805, 0.036915353095352356, 0.10890029163128945,
           0.050204880209679205, 0.034700431909631214, 0.04097604193584112, 0.28092583705563146, 0.2517627081103031,
           0.007752224150023995, 0.04872826608586511, 0.0014766141238140942, 0.006275610026209901, 0.01439698770718742,
           0.0007383070619070471]


def norm_data(csv, out):
    with open(csv, 'r') as f:
        data = json.load(f)
    with open(out, 'w') as f:
        for each in data:
            f.write(each)
            for tag in static_tags_name:
                if tag in data[each]:
                    f.write(' 1')
                else:
                    f.write(' 0')
            f.write('\n')


def get_weights(csv):
    with open(csv, 'r') as f:
        data = json.load(f)
    dict = {}
    total = 0
    for each in static_tags_name:
        dict[each] = 0
    for each in data:
        for tag in data[each]:
            dict[tag] += 1
            total += 1
    weights = []
    for each in static_tags_name:
        weights.append(dict[each] * 10 / total)
    print(total)
    print(dict)
    print(weights)


if __name__ == "__main__":
    norm_data('train_data.json', 'train_data.txt')
    # get_weights('train_data.json')
