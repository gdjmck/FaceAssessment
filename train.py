from utils.log import Train_Log
from utils.loss import hinge_loss, accuracy, bce_loss
from utils.arguments import get_args
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam
import numpy as np
import model
import dataset
from torchvision import transforms

val_split = 0.1
shuffle_dataset = True
args = get_args()
logger = Train_Log(args)

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
    start_epoch = 0
    if args.resume:
        start_epoch, model_assess = logger.load_model(model_assess)
    device = torch.device('cpu')
    if args.gpu:
        if torch.cuda.is_available():
            device = torch.device('cuda')
            model_assess.to(device)
        else:
            print('NO GPU AVAILABLE, USE CPU INSTEAD.')

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(115., 98., 87.6), std=(128, 128, 128))])
    
    data = dataset.AssessSet(args.dataDir, transform=transform)
    train_sampler, val_sampler = gen_split_sampler(data)
    train_loader = DataLoader(dataset=data,
                            batch_size=1, shuffle=False, 
                            num_workers=4, drop_last=False,
                            sampler=train_sampler,
                            collate_fn=dataset.my_collate)
    val_loader = DataLoader(dataset=data, 
                            batch_size=1, shuffle=False,
                            num_workers=1, drop_last=False,
                            sampler=val_sampler,
                            collate_fn=dataset.my_collate)
    
    extractor_params = model_assess.extractor.parameters()
    optimizer = Adam([
        {'params': [p for p in model_assess.parameters() if p not in extractor_params]},
        {'params': model_assess.extractor.parameters(), 'lr':0.1*args.lr}
    ], lr=args.lr)

    best_val_acc, best_val_loss = 0., 10
    for epoch in range(start_epoch, start_epoch+args.epochs):
        loss_, acc_ = 0., 0.
        model_assess.train()
        for i, sample in enumerate(train_loader):
            img = sample['img'].to(device)
            #print('batch shape:', img.shape)
            score = sample['score'].to(device)

            score_pred = model_assess(img)
            loss = bce_loss(score_pred, score)
            acc_ += accuracy(score_pred, score)
            loss_ += loss.item()
            print('sample %d:\tloss:%.4f'% (i, loss.mean()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.step()
        print('epoch %d acc: %.4f\t loss: %.4f' %(epoch, acc_/(i+1), loss_/(i+1)))

        # eval on valid set
        loss_val_, acc_val_ = 0., 0.
        model_assess.eval()
        for j, sample in enumerate(val_loader):
            img = sample['img'].to(device)
            score = sample['score'].to(device)
            score_pred = model_assess(img)
            loss_val_ += bce_loss(score_pred, score).item()
            acc_val_ += accuracy(score_pred, score)
        avg_loss, avg_acc = loss_val_/(j+1), acc_val_/(j+1)
        print('\teval acc: %.4f\t loss: %.4f' %(avg_acc, avg_loss))
        if avg_loss <= best_val_loss and avg_acc > best_val_acc:
            best_val_loss = avg_loss
            best_val_acc = avg_acc
            logger.save_model(model_assess, epoch)
            print('save best model at epoch ', epoch)

        

if __name__ == '__main__':
    main()