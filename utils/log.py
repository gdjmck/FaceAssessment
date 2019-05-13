from tensorboardX import SummaryWriter
import os
import torch

class Train_Log():
    def __init__(self, args):
        self.args = args

        self.summary = SummaryWriter(args.saveDir)
        self.step_cnt = 1
        if not os.path.exists(args.saveDir):
            os.makedirs(args.saveDir)

        self.save_dir_model = os.path.join(args.saveDir, 'model')
        if not os.path.exists(self.save_dir_model):
            os.makedirs(self.save_dir_model)

        if os.path.exists(self.save_dir + '/log.txt'):
            self.logFile = open(self.save_dir + '/log.txt', 'a')
        else:
            self.logFile = open(self.save_dir + '/log.txt', 'w')

    def add_scalar(self, scalar_name, scalar, step=None):
        if step is None:
            step = self.step_cnt
        self.summary.add_scalar(scalar_name, scalar, step)
        
    def add_histogram(self, var_name, value, step=None):
        if step is None:
            step = self.step_cnt
        self.summary.add_histogram(var_name, value, step)
 
    def add_image(self, tag, image):
        if isinstance(image, torch.autograd.Variable):
            image = image.data
        if len(image.shape) == 4:
            image = image[0, :, :, :]
        image = image.cpu().numpy()
        self.summary.add_image(tag, image, self.step_cnt)
        
    def step(self):
        self.step_cnt += 1

    def save_model(self, model, epoch):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'step': self.step_cnt
        }, lastest_out_path)

        model_out_path = "{}/model_obj.pth".format(self.save_dir_model)
        torch.save(
            model,
            model_out_path)

    def load_model(self, model):

        lastest_out_path = "{}/ckpt_lastest.pth".format(self.save_dir_model)
        if self.args.without_gpu: # 用cpu载入模型到内存
            ckpt = torch.load(lastest_out_path, map_location='cpu')
        else: # 模型载入到显存
            ckpt = torch.load(lastest_out_path)
        state_dict = ckpt['state_dict'].copy()
        for key in ckpt['state_dict']:
            if key not in model.state_dict():
                print('missing key:\t', key)
                state_dict.pop(key)
        ckpt['state_dict'] = state_dict
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'], strict=False)
        self.step_cnt = ckpt['step']
        #self.step_cnt = 1
        print("=> loaded checkpoint '{}' (epoch {}  total step {})".format(lastest_out_path, ckpt['epoch'], self.step_cnt))

        return start_epoch, model


    def save_log(self, log):
        self.logFile.write(log + '\n')