import pandas as pd

class Evaluator:
    def __init__(self, hyp_pth):
        hyp_d = pd.read_csv(hyp_pth).drop(['Article'], axis=1).dropna()
        hyp_d['Hypothesis path'] = hyp_d['Hypothesis path'].apply(lambda x: x.replace('/', ', '))
        hyp_d['Hypothesis path'] = hyp_d['Hypothesis path'].apply(lambda x: x.replace(';', ','))
        hyp_d['Hypothesis path'] = hyp_d['Hypothesis path'].apply(lambda x: x.split(', '))

        hyp_d['EFO ID'] = hyp_d['EFO ID'].apply(lambda x: x.replace(':', '_'))
        hyp_d['EFO ID'] = hyp_d['EFO ID'].apply(lambda x: x.split(', '))

        gt_pths = []

        for i in range(len(hyp_d)):
            lin = hyp_d.iloc[i]
            diseases = lin['EFO ID']
            h_pth = lin['Hypothesis path'][::-1]
            for d in diseases:
                gt_pths.append(
                    tuple([d] + h_pth)
                )
        self.gt_pths = gt_pths

    @staticmethod
    def lev_dist(s1, s2):
        if len(s1) > len(s2):
            s1, s2 = s2, s1

        distances = range(len(s1) + 1)
        for i2, c2 in enumerate(s2):
            distances_ = [i2+1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    distances_.append(distances[i1])
                else:
                    distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
            distances = distances_
        return distances[-1]

    def evaluate(self, G, node_id_dict, soft_yen_func, penalty_coef):
        # evaluation (TODO: review later)
        w0 = G.es['weight'].copy()
        s = 0
        num_good = 0
        pth_dists = []
        all_preds = []

        for gt_pth in self.gt_pths:
            try:
                gt_nums = [node_id_dict[el] for el in gt_pth]
            except:
                continue
            id1  = gt_nums[0]
            id2 = gt_nums[-1]
            G.es['weight'] = w0.copy()
            pr = soft_yen_func(G, id1, id2, penalty_coef=penalty_coef)[0]
            pred_pths = list(set([tuple(p) for p in pr]))

            all_preds += pred_pths

            dists = [
                self.lev_dist(gt_nums, pred_pth) for pred_pth in pred_pths
            ]
            pth_dist = sum(dists) / len(dists)
            # pth_dist = min(dists)
            pth_dists.append(pth_dist)
        G.es['weight'] = w0.copy()
        return pth_dists, all_preds