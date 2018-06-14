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
        static_tags_name = ['cardiac monitor', 'lymphatic diseases', 'pulmonary disease', 'osteophytes', 'foreign body',
                            'dish', 'aorta, thoracic', 'atherosclerosis', 'histoplasmosis', 'hypoventilation',
                            'catheterization, central venous', 'pleural effusions', 'pleural effusion', 'callus',
                            'sternotomy', 'lymph nodes', 'tortuous aorta', 'stent', 'interstitial pulmonary edema',
                            'cholecystectomies', 'neoplasm', 'central venous catheter', 'pneumothorax',
                            'metastatic disease', 'vena cava, superior', 'cholecystectomy', 'scoliosis',
                            'subcutaneous emphysema', 'thoracolumbar scoliosis', 'spinal osteophytosis',
                            'pulmonary fibroses', 'rib fractures', 'sarcoidosis', 'eventration', 'fibrosis', 'spine',
                            'obstructive lung disease', 'pneumonitis', 'osteopenia', 'air trapping', 'demineralization',
                            'mass lesion', 'pulmonary hypertension', 'pleural diseases', 'pleural thickening',
                            'calcifications of the aorta', 'calcinosis', 'cystic fibrosis', 'empyema', 'catheter',
                            'lymph', 'pericardial effusion', 'lung cancer', 'rib fracture', 'granulomatous disease',
                            'chronic obstructive pulmonary disease', 'rib', 'clip', 'aortic ectasia', 'shoulder',
                            'scarring', 'scleroses', 'adenopathy', 'emphysemas', 'pneumonectomy', 'infection',
                            'aspiration', 'bilateral pleural effusion', 'bulla', 'lumbar vertebrae', 'lung neoplasms',
                            'lymphadenopathy', 'hyperexpansion', 'ectasia', 'bronchiectasis', 'nodule', 'pneumonia',
                            'right-sided pleural effusion', 'osteoarthritis', 'thoracic spondylosis', 'picc',
                            'cervical fusion', 'tracheostomies', 'fusion', 'thoracic vertebrae', 'catheters',
                            'emphysema', 'trachea', 'surgery', 'cervical spine fusion', 'hypertension, pulmonary',
                            'pneumoperitoneum', 'scar', 'atheroscleroses', 'aortic calcifications', 'volume overload',
                            'right upper lobe pneumonia', 'apical granuloma', 'diaphragms', 'copd', 'kyphoses',
                            'spinal fractures', 'fracture', 'clavicle', 'focal atelectasis', 'collapse',
                            'thoracotomies', 'congestive heart failure', 'calcified lymph nodes', 'edema',
                            'degenerative disc diseases', 'cervical vertebrae', 'diaphragm', 'humerus', 'heart failure',
                            'normal', 'coronary artery bypass', 'pulmonary atelectasis', 'lung diseases, interstitial',
                            'pulmonary disease, chronic obstructive', 'opacity', 'deformity', 'chronic disease',
                            'pleura', 'aorta', 'tuberculoses', 'hiatal hernia', 'scolioses', 'pleural fluid',
                            'malignancy', 'kyphosis', 'bronchiectases', 'congestion', 'discoid atelectasis', 'nipple',
                            'bronchitis', 'pulmonary artery', 'cardiomegaly', 'thoracic aorta', 'arthritic changes',
                            'pulmonary edema', 'vascular calcification', 'sclerotic', 'central venous catheters',
                            'catheterization', 'hydropneumothorax', 'aortic valve', 'hyperinflation', 'prostheses',
                            'pacemaker, artificial', 'bypass grafts', 'pulmonary fibrosis', 'multiple myeloma',
                            'postoperative period', 'cabg', 'right lower lobe pneumonia', 'granuloma',
                            'degenerative change', 'atelectasis', 'inflammation', 'effusion', 'cicatrix',
                            'tracheostomy', 'aortic diseases', 'sarcoidoses', 'granulomas', 'interstitial lung disease',
                            'infiltrates', 'displaced fractures', 'chronic lung disease', 'picc line',
                            'intubation, gastrointestinal', 'lung diseases', 'multiple pulmonary nodules',
                            'intervertebral disc degeneration', 'pulmonary emphysema', 'spine curvature', 'fibroses',
                            'chronic granulomatous disease', 'degenerative disease', 'atelectases', 'ribs',
                            'pulmonary arterial hypertension', 'edemas', 'pectus excavatum', 'lung granuloma',
                            'plate-like atelectasis', 'enlarged heart', 'hilar calcification', 'heart valve prosthesis',
                            'tuberculosis', 'old injury', 'patchy atelectasis', 'histoplasmoses', 'exostoses',
                            'mastectomies', 'right atrium', 'large hiatal hernia', 'hernia, hiatal', 'aortic aneurysm',
                            'lobectomy', 'spinal fusion', 'spondylosis', 'ascending aorta', 'granulomatous infection',
                            'fractures, bone', 'calcified granuloma', 'degenerative joint disease',
                            'intubation, intratracheal', 'others']

        return static_tags_name
