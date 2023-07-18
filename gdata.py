import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from torch_geometric.data import Data
import numpy as np

class GData:
    def __init__(self, nodes_pth, edges_pth, types_of_interest, emb_dim, dummy_emb):
        if dummy_emb:
            self.nodes_full = pd.read_csv(nodes_pth, index_col=0)
            # self.nodes_full['attributes'] = self.nodes_full['attributes']
        else:
            self.nodes_full = pd.read_csv(nodes_pth, index_col=0).dropna()
            self.nodes_full['attributes'] = self.nodes_full['attributes'].apply(eval)

        self.mid_emb_idx = list(self.nodes_full[~self.nodes_full['attributes'].isna()].index)

        self.edges_full = pd.read_csv(edges_pth, index_col=0)
        self.edges_full = self.edges_full.merge(self.nodes_full,
                                                left_on='subj',
                                                right_on='id',
                                                how='inner').merge(self.nodes_full,
                                                                   left_on='obj',
                                                                   right_on='id',
                                                                   how='inner')[['subj', 'obj']]
        self.types_of_interest = types_of_interest
        self.nodes_full = self.nodes_full[self.nodes_full['type'].apply(lambda x: x in types_of_interest)]

        self.node_id_dict = {}
        self.k = len(self.nodes_full)
        for i in range(self.k):
            self.node_id_dict[self.nodes_full.iloc[i]['id']] = i

        self.edges_list = []
        for i in range(len(self.edges_full)):
            s_id = self.edges_full.iloc[i]['subj']
            o_id = self.edges_full.iloc[i]['obj']
            try:
                self.edges_list.append((self.node_id_dict[s_id], self.node_id_dict[o_id]))
            except:
                continue

        self.n = len(self.edges_list)

        self.x = self.get_x(emb_dim, dummy_emb)

    def get_x(self, dim, dummy=False):
        nf = self.nodes_full.copy()

        nf = nf[~nf['attributes'].isna()]
        nf['attributes'] = nf['attributes'].apply(eval)
        tps = list(nf['type'].unique())

        base_feats = {}
        for t in tps:
            ks = nf[nf['type'] == t].iloc[0]['attributes']
            for ky in ks:
                base_feats[ky] = nf[nf['type'] == t].apply('attributes').apply(lambda x: x[ky]).unique()
                base_feats[ky].sort()
        for b_f in base_feats:
            for v in base_feats[b_f]:
                nf[b_f + '_' + str(v)] = 0
        for i in range(nf.index[-1] + 1):
            try:
                attrs = nf.loc[nf.index==i]['attributes'].item()
            except:
                continue

            for ky in attrs:
                nf.loc[nf.index==i, ky + '_' + str(attrs[ky])] = 1

        feats = nf.drop(['id', 'name', 'type', 'attributes'], axis=1).values
        scaler = StandardScaler()
        feats_t = scaler.fit_transform(feats.copy())

        # Use PCA to reduce dimensionality of the matrix.
        d = feats_t.shape[1]
        emb_d = dim
        pca = PCA(emb_d)
        feats_t = pca.fit_transform(feats_t)

        if dummy:
            self.mid_emb = torch.tensor(feats_t, dtype=torch.float)
            return torch.tensor(pd.get_dummies(self.nodes_full['type']).values, dtype=torch.float)

        return torch.tensor(feats_t, dtype=torch.float)

    def get_tg(self, keep_rate):
        e_idx = torch.tensor(self.edges_list).t().contiguous()
        e_lab_idx = e_idx[:,np.random.choice(self.n, (int(self.n * keep_rate),), replace=False)]
        e_lab = self.x.new_ones(size=(int(self.n * keep_rate),))
        ds = Data(
            x=self.x, edge_index=e_idx, edge_label=e_lab, edge_label_index=e_lab_idx, mid_emb_idx=self.mid_emb_idx, mid_emb=self.mid_emb
        )
        return ds