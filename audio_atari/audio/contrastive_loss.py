import torch
from torch import nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # self.register_buffer("temperature", torch.tensor(temperature).to(device=device))
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())
        # self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=int)).float().to(device=device))
        self.temperature = torch.tensor(temperature).to(device=device)
        self.negatives_mask = 1-torch.eye(batch_size * 2,batch_size * 2).float().to(device=device)
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 
        """
        # z_i = F.normalize(emb_i, dim=1)
        # z_j = F.normalize(emb_j, dim=1)
        # print(emb_i, z_i)

        representations = torch.cat([emb_i, emb_j], dim=0)
        # print(representations.shape) # torch.Size([64, 1]) for batch size 32

        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)

        similarity_matrix = similarity_matrix.squeeze()
        # print('similarity matrix: ',similarity_matrix) 

        # print(similarity_matrix.is_cuda)
        # print(similarity_matrix.shape) #torch.Size([64, 64])
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        nominator = torch.exp(positives / self.temperature)
        # print('numerator:',nominator.shape)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        # print('negatives mask:', self.negatives_mask)
        # print('denominator sum:',torch.sum(denominator, dim=1).shape)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # print('loss partial: ',loss_partial)
        # print('loss_partial: ',loss_partial.shape)
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        print('loss:',loss)
        return loss


class ContrastiveSingleLoss(nn.Module):
    # only uses the loss for the first sample compared with all other samples
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = torch.tensor(temperature).to(device=device)
        self.negatives_mask = 1-torch.eye(batch_size * 2,batch_size * 2).float().to(device=device)
            
    def forward(self, emb_i, emb_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 
        """

        representations = torch.cat([emb_i, emb_j], dim=0) # torch.Size([64, 1]) for batch size 32

        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)
        similarity_matrix = similarity_matrix.squeeze() #torch.Size([64, 64])
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        # only taking loss of the first element against all other elements in the batch (Yes v/s No's or No v/s Yes's)
        loss = (loss_partial[0]+loss_partial[self.batch_size]) / (2)# * self.batch_size)
        print('loss:',loss)
        return loss



class ContrastiveSingleProsodyLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = torch.tensor(temperature).to(device=device)
        self.negatives_mask = 1-torch.eye(batch_size * 2,batch_size * 2).float().to(device=device)
            
    def forward(self, emb_i, emb_j, prosody_i, prosody_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 
        """

        representations = torch.cat([emb_i, emb_j], dim=0) # torch.Size([64, 1]) for batch size 32

        eps = 0.01
        prosody_diff = abs(prosody_i-prosody_j)
        

        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)
        similarity_matrix = similarity_matrix.squeeze() 
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # set temperature as the difference between overall energy of utterances. 
        # Ensure temp between 0 and 1 (softmax all energy differences in this batch). 
        # Are rewards scaled? No. And here we consider returns which could be arbitrarily long.
        # a higher temperature is scaling down the similairity of rewards and lower temperature is scaling up the similarity of rewards
        
        softmax = nn.Softmax(dim=0) #TODO: check dimension correct based on prosody_diff.shape?
        prosody_diff_softmax = softmax(prosody_diff)
        prosody = torch.cat([prosody_diff_softmax, prosody_diff_softmax], dim=0) +eps
        
        # nominator = torch.exp(positives / self.temperature)
        nominator = torch.div(positives, prosody)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        print('loss:',loss)
        return loss


class ContrastivePASELoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = torch.tensor(temperature).to(device=device)
        self.negatives_mask = 1-torch.eye(batch_size * 2,batch_size * 2).float().to(device=device)
            
    def forward(self, emb_i, emb_j, pase_i, pase_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 
        """

        representations = torch.cat([emb_i, emb_j], dim=0) # torch.Size([64, 1]) for batch size 32

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # print('in CAL: ',pase_i.shape,pase_j.shape)
        pase_cosine = cos(pase_i, pase_j) # TODO: check shape is same as batch size
        #min_pase_cos, max_pase_cos = torch.min(pase_cosine), torch.max(pase_cosine)
        min_cos, max_cos = -1,1
        pase_cosine_scaled = (pase_cosine-min_cos)/(max_cos-min_cos)
        pase_cosine_scaled_rep = torch.cat([pase_cosine_scaled, pase_cosine_scaled], dim=0)

        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)

        similarity_matrix = similarity_matrix.squeeze()
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
               
        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
    
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        # a weighted version of similarity between utterances is used to scale each term in the numerator
        # based on how close the pair actually is (from cosine similarity of their audio embeddings)
        # construct a dummy example
        # scale the loss by the cosine similarity (converted between 0,1 from -1,1) of the audio embeddings
        loss = torch.sum(pase_cosine_scaled_rep * loss_partial) / (2 * self.batch_size)
        print('loss:',loss)
        return loss


class ContrastivePASEProsodyLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super().__init__()
        self.batch_size = batch_size
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.temperature = torch.tensor(temperature).to(device=device)
        self.negatives_mask = 1-torch.eye(batch_size * 2,batch_size * 2).float().to(device=device)
            
    def forward(self, emb_i, emb_j, pase_i, pase_j, prosody_i, prosody_j):
        """
        emb_i and emb_j are batches of embeddings, where corresponding indices are pairs
        z_i, z_j as per SimCLR paper
        https://zablo.net/blog/post/understanding-implementing-simclr-guide-eli5-pytorch/ 
        """

        representations = torch.cat([emb_i, emb_j], dim=0) # torch.Size([64, 1])

        eps = 0.01
        prosody_diff = abs(prosody_i-prosody_j)

        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        pase_cosine = cos(pase_i, pase_j) # TODO: check shape is same as batch size
        min_cos, max_cos = -1,1
        pase_cosine_scaled = (pase_cosine-min_cos)/(max_cos-min_cos)
        pase_cosine_scaled_rep = torch.cat([pase_cosine_scaled, pase_cosine_scaled], dim=0)

        # compute similarity based on difference of rewards
        r1 = representations.unsqueeze(0).repeat(self.batch_size*2,1,1)
        r2 = representations.unsqueeze(1).repeat(1,self.batch_size*2,1)
        sim = torch.abs(r1-r2)

        # change L2 distance to similarity by subtracting by 1? use 1/(1+d(p1,p2))
        similarity_matrix = 1/(1+sim)

        similarity_matrix = similarity_matrix.squeeze()
        
        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        
        # set temperature as the difference between overall energy of utterances. 
        # Ensure temp between 0 and 1 (softmax all energy differences in this batch). 
        # Are rewards scaled? No. And here we consider returns which could be arbitrarily long.
        # a higher temperature is scaling down the similairity of rewards and lower temperature is scaling up the similarity of rewards
        softmax = nn.Softmax(dim=0) #TODO: check dimension correct based on prosody_diff.shape?
        prosody_diff_softmax = softmax(prosody_diff)
        prosody = torch.cat([prosody_diff_softmax, prosody_diff_softmax], dim=0) +eps
        
        # nominator = torch.exp(positives / self.temperature)
        nominator = torch.div(positives, prosody)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))

        # a weighted version of similarity between utterances is used to scale each term in the numerator
        # based on how close the pair actually is (from cosine similarity of their audio embeddings)
        # construct a dummy example
        # scale the loss by the cosine similarity (converted between 0,1 from -1,1) of the audio embeddings
        loss = torch.sum(pase_cosine_scaled_rep * loss_partial) / (2 * self.batch_size)
        print('loss:',loss)
        return loss