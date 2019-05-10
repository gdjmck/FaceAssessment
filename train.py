import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import numpy as np
import model
import dataset
from utils.loss import hinge_loss

def main():
    model_assess = model.FaceAssess()
    dataloader = DataLoader(dataset=dataset.AssessSet('high-res'),
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        

if __name__ == '__main__':
    main()