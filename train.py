import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import model
import dataset
from torchvision import transforms
from utils.loss import hinge_loss

def main():
    model_assess = model.FaceAssess()

    transform = transforms.Compose([transforms.Resize(512),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=(115., 98., 87.6), std=(1, 1, 1))])
    dataloader = DataLoader(dataset=dataset.AssessSet('high-res', transform=transform),
                            batch_size=1, shuffle=True, 
                            num_workers=4, drop_last=False)
    
    model_assess.train()
    extractor_params = model_assess.extractor.parameters()
    optimizer = Adam([
        {'params': [p for p in model_assess.parameters() if p not in extractor_params]},
        {'params': model_assess.extractor.parameters(), 'lr':1e-5}
    ], lr=1e-4)

    for epoch in range(100):
        for sample in dataloader:
            img = sample['img']
            score = sample['score']

            score_pred = model_assess(img)
            loss = hinge_loss(score_pred, score)
            print('loss:', loss.item(), '\tscore_pred:', score_pred.item(), '\tscore_gt:', score.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

if __name__ == '__main__':
    main()