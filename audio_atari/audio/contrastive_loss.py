import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.register_buffer("temperature", torch.tensor(temperature).to(device=device))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=int)).float().to(device=device))

            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)
        # print(emb_i, z_i)

        representations = torch.cat([emb_i, emb_j], dim=0)
        # print(representations.shape) # torch.Size([64, 1])


        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)

        similarity_matrix = similarity_matrix.squeeze()

        # print(similarity_matrix.is_cuda)
        # print(similarity_matrix.shape) #torch.Size([64, 64])
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # TODO: set temperature as the difference between overall energy of utterances. 
        # Ensure temp between 0 and 1 (softmax all energy differences in this batch). 
        # Are rewards scaled? No. And here we consider returns which could be arbitrarily long.
        # a higher temperature is scaling down the similairity of rewards and lower temperature is scaling up the similarity of rewards
        
        # TODO: a weighted version of similarity between utterances is used to scale each term in the numerator
        # based on how close the pair actually is (from cosine similarity of their audio embeddings)
        # construct a dummy example
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss