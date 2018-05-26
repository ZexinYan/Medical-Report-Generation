class Tag(object):
    def __init__(self):
        self.static_tags = self.__load_static_tags()
        self.id2tags = self.__load_id2tags()
        self.tags2id = self.__load_tags2id()

    def array2tags(self, array):
        tags = []
        for id in array:
            tags.append(self.id2tags[id])
        return tags

    def tags2array(self, tags):
        array = []
        for tag in self.static_tags:
            if tag in tags:
                array.append(1)
            else:
                array.append(0)
        return array

    def inv_tags2array(self, array):
        tags = []
        for i, value in enumerate(array):
            if value != 0:
                tags.append(self.id2tags[i])
        return tags

    def __load_id2tags(self):
        id2tags = {}
        for i, tag in enumerate(self.static_tags):
            id2tags[i] = tag
        return id2tags

    def __load_tags2id(self):
        tags2id = {}
        for i, tag in enumerate(self.static_tags):
            tags2id[tag] = i
        return tags2id

    def __load_static_tags(self):
        static_tags_name = ['nodule', 'borderline', 'spinal fusion', 'cardiac shadow', 'interstitial',
                            'pulmonary congestion',
                            'technical quality of image unsatisfactory', 'bronchiectasis', 'cervical vertebrae',
                            'hypoinflation', 'medical device', 'prominent', 'mass', 'breast implants', 'calcinosis',
                            'aortic aneurysm', 'aorta, thoracic', 'lower lobe', 'scattered', 'left',
                            'lung, hyperlucent',
                            'pneumoperitoneum', 'enlarged', 'foreign bodies', 'epicardial fat', 'reticular', 'abnormal',
                            'irregular', 'obscured', 'large', 'diaphragm', 'right', 'breast', 'pulmonary edema',
                            'hyperostosis, diffuse idiopathic skeletal', 'airspace disease', 'stents', 'mild',
                            'volume loss',
                            'shift', 'sulcus', 'humerus', 'lucency', 'blunted', 'osteophyte', 'blood vessels',
                            'lumbar vertebrae', 'flattened', 'tortuous', 'small', 'healed', 'hypertension, pulmonary',
                            'bone diseases, metabolic', 'trachea', 'atherosclerosis', 'mediastinum', 'coronary vessels',
                            'lung',
                            'chronic', 'multiple', 'ribs', 'pulmonary disease, chronic obstructive', 'apex', 'hilum',
                            'spondylosis', 'diffuse', 'paratracheal', 'pneumothorax', 'clavicle', 'retrocardiac',
                            'lymph nodes',
                            'bronchovascular', 'azygos lobe', 'pulmonary emphysema', 'granulomatous disease',
                            'calcified granuloma', 'normal', 'thoracic vertebrae', 'funnel chest', 'thorax', 'aorta',
                            'adipose tissue', 'anterior', 'arthritis', 'emphysema', 'fractures, bone', 'hernia, hiatal',
                            'implanted medical device', 'sutures', 'granuloma', 'pleura', 'thickening', 'cysts',
                            'upper lobe',
                            'middle lobe', 'pleural effusion', 'deformity', 'contrast media', 'pulmonary atelectasis',
                            'hyperdistention', 'pericardial effusion', 'spine', 'mastectomy', 'surgical instruments',
                            'nipple shadow', 'heart', 'streaky', 'blister', 'catheters, indwelling', 'bilateral',
                            'neck',
                            'cavitation', 'density', 'scoliosis', 'pulmonary artery', 'round', 'opacity',
                            'lung diseases, interstitial', 'sternum', 'heart ventricles', 'lingula', 'aortic valve',
                            'heart failure', 'heart atria', 'sarcoidosis', 'bullous emphysema', 'sclerosis',
                            'costophrenic angle', 'kyphosis', 'hydropneumothorax', 'consolidation', 'dislocations',
                            'markings',
                            'abdomen', 'tube, inserted', 'no indexing', 'pneumonectomy', 'posterior', 'patchy',
                            'diaphragmatic eventration', 'pulmonary fibrosis', 'pneumonia', 'cardiomegaly', 'focal',
                            'cicatrix',
                            'elevated', 'infiltrate', 'moderate', 'degenerative', 'base', 'trachea, carina', 'severe',
                            'bronchi', 'pulmonary alveoli', 'shoulder', 'cystic fibrosis']
        return static_tags_name
