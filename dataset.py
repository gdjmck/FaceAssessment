import os
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image

def get_file_path(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))
    return tensor

def brighten(im_bgr, ratio=1.1):
    img_hsv = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[..., -1] *= ratio
    return cv2.cvtColor(np.clip(img_hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

def my_collate(batch):
    data = [item['img'] for item in batch]
    label = [item['score'] for item in batch]
    return torch.stack(data, 0), torch.stack(label, 0)

class AssessSet(data.Dataset):
    '''
        #TODO: 添加最大尺寸的scale
    '''
    def __init__(self, root_folder, transform=None):
        super(AssessSet, self).__init__()
        self.transform = transform
        '''
        self.degree_table = {'0': 1,
                            '1': 0,
                            '2': 0,
                            '3': 0,
                            '4': 0}
        '''
        self.num_degree = len(os.listdir(root_folder))
        self.img_files = []
        self.img_degree = []

        full_paths = get_file_path(os.path.join(root_folder, 'blur0'))
        self.img_files += full_paths

    def __len__(self):
        return len(self.img_files)

    '''
    def degree_to_score(self, degree):
        return self.degree_table[str(degree)]
    '''

    def __getitem__(self, index):
        img = Image.open(self.img_files[index])
        img = cv2.imread(self.img_files[index], -1)
        img_b = brighten(img, ratio=np.random.randint(90, 110)/100)
        good = [img, cv2.flip(img, 1), img_b, cv2.flip(img_b, 1)]
        bad = [Image.open(self.img_files[index].replace('blur0', 'blur'+str(i))) for i in range(1, 5)]
        if self.transform is not None:
            good = [self.transform(Image.fromarray(item)) for item in good]
            bad = [self.transform(item) for item in bad]
        #score = torch.FloatTensor([self.degree_to_score(self.img_degree[index])])
        return [{'img': item, 'score': torch.FloatTensor([1])} for item in good] + 
                [{'img': item, 'score': torch.FloatTensor([0])} for item in bad]
