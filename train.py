import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import numpy as np
import model
import dataset
from torchvision import transforms
from utils.loss import hinge_loss, accuracy

val_split = 0.2
shuffle_dataset = True

def gen_split_sampler(dataset):
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    if shuffle_dataset:
        np.random.seed(42)
        np.random.shuffle(indices)
    val_size = int(np.floor(dataset_size * val_split))
    train_indices, val_indices = indices[val_size:], indices[:val_size]
    return SubsetRandomSampler(train_indices), SubsetRandomSampler(val_indices)

def main():
    model_assess = model.FaceAssess()

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(115., 98., 87.6), std=(128, 128, 128))])
    
    data = dataset.AssessSet('high-res', transform=transform)
    train_sampler, val_sampler = gen_split_sampler(data)
    train_loader = DataLoader(dataset=data,
                            batch_size=1, shuffle=False, 
                            num_workers=4, drop_last=False,
                            sampler=train_sampler)
    val_loader = DataLoader(dataset=data, 
                            batch_size=1, shuffle=False,
                            num_workers=1, drop_last=False,
                            sampler=val_sampler)
    
    extractor_params = model_assess.extractor.parameters()
    optimizer = Adam([
        {'params': [p for p in model_assess.parameters() if p not in extractor_params]},
        {'params': model_assess.extractor.parameters(), 'lr':1e-5}
    ], lr=1e-4)

    for epoch in range(100):
        loss_, acc_ = 0., 0.

        model_assess.train()
        for i, sample in enumerate(train_loader):
            if i > 10:
                break
            img = sample['img']
            score = sample['score']

            score_pred = model_assess(img)
            loss = hinge_loss(score_pred, score)
            acc_ += accuracy(score_pred, score)
            loss_ += loss.item()
            print('loss:%.4f \tscore_pred:%.4f \tscore_gt:%.4f'% (loss.item(), score_pred.item(), score.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch %d acc: %.4f\t loss: %.4f' %(epoch, acc_/len(train_loader, loss_/len(train_loader))))

        # eval on valid set
        loss_val_, acc_val_ = 0., 0.
        model_assess.eval()
        for j, sample in enumerate(val_loader):
            if j > 5:
                break
            img = sample['img']
            score = sample['score']
            score_pred = model_assess(img)
            loss_val_ += hinge_loss(score_pred, score).item()
            acc_val_ += accuracy(score_pred, score)
        print('\teval acc: %.4f\t loss: %.4f' %(acc_val_/len(val_loader), loss_val_/len(val_loader)))

        

if __name__ == '__main__':
    main()