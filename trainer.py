import torch
from torch_geometric.utils import negative_sampling

class Trainer:
    def __init__(self, model, optimizer, criterion_emb, criterion_cls, neg2pos_ratio, ds, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion_emb = criterion_emb
        self.criterion_cls = criterion_cls
        self.device = device
        self.neg2pos = neg2pos_ratio
        self.ds = ds

    def train(self, n_epochs, num_neg_eval):
        for ep in range(n_epochs):
            self.model.train()
            self.optimizer.zero_grad()

            out_emb = self.model.encode_1(self.ds.x.to(self.device), self.ds.edge_index.to(self.device))
            loss_emb = self.criterion_emb(out_emb[self.ds.mid_emb_idx], self.ds.mid_emb.to(self.device))

            z = self.model.encode_2(out_emb, self.ds.edge_index.to(self.device))

            neg_edge_index = negative_sampling(
                edge_index=self.ds.edge_index, num_nodes=self.ds.num_nodes,
                num_neg_samples=int(self.ds.edge_label_index.size(1) * self.neg2pos), method='sparse')

            edge_label_index = torch.cat(
                [self.ds.edge_label_index, neg_edge_index],
                dim=-1,
            )

            edge_label = torch.cat([
                self.ds.edge_label,
                self.ds.edge_label.new_zeros(neg_edge_index.size(1))
            ], dim=0)

            out = self.model.decode(z, edge_label_index.to(self.device)).view(-1)
            loss_cls = self.criterion_cls(out, edge_label.to(self.device))
            loss = loss_cls + loss_emb * 0.1
            loss.backward()
            self.optimizer.step()

            if ep % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    z = self.model.encode_1(self.ds.x.to(self.device), self.ds.edge_index.to(self.device))
                    z = self.model.encode_2(z, self.ds.edge_index.to(self.device))
                    tp_p = (self.model.decode(z, self.ds.edge_index.to(self.device)) > 0.0).float().mean().item()

                    neg_edge_index = negative_sampling(
                        edge_index=self.ds.edge_index, num_nodes=self.ds.num_nodes,
                        num_neg_samples=num_neg_eval, method='sparse')
                    tn_n = (self.model.decode(z, neg_edge_index.to(self.device)) < 0.0).float().mean().item()
                print('epoch:', ep, '|| losses:', loss_emb.item(), loss_cls.item(), '|| prec:', tp_p, '|| spec:', tn_n)

    def predict_scores(self):
        with torch.no_grad():
            self.model.eval()
            z = self.model.encode_1(self.ds.x.to(self.device), self.ds.edge_index.to(self.device))
            z = self.model.encode_2(z, self.ds.edge_index.to(self.device))
            scores = self.model.decode(z, self.ds.edge_index.to(self.device)).detach().cpu().sigmoid()
            print('precision:', (self.model.decode(z, self.ds.edge_index.to(self.device)) > 0.0).float().mean().item())
        return scores