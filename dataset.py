import os
import torch
import torch.utils.data as data
from skimage import io

def get_file_path(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder)]

def np2Tensor(array):
    ts = (2, 0, 1)
    tmp = array.copy()
    tensor = torch.FloatTensor(tmp.transpose(ts).astype(float))
    return tensor

class AssessSet(data.Dataset):
    def __init__(self, root_folder):
        super(AssessSet, self).__init__()
        self.degree_table = {'0': 1.0,
                            '1': 0.8,
                            '2': 0.6,
                            '3': 0.4,
                            '4': 0.2}
        self.num_degree = len(os.listdir(root_folder))
        self.img_files = []
        self.img_degree = []

        for i in range(self.num_degree):
            full_paths = get_file_path(os.path.join(root_folder, 'blur'+str(i)))
            self.img_files += full_paths
            self.img_degree += [i] * len(full_paths)

    def __len__(self):
        return len(self.img_files)

    def degree_to_score(self, degree):
        return self.degree_table[str(degree)]

    def __getitem__(self, index):
        img = io.imread(self.img_files[index]).astype('float')
        img -= (115., 98., 87.6)
        img = np2Tensor(img)
        score = torch.FloatTensor([self.degree_to_score(self.img_degree[index])])
        print('img', img.shape, 'score', type(score))
        return {'img': img, 'score': score}