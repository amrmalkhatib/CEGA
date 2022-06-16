from tqdm import tqdm
import sys, math
import numpy as np
from multiprocessing import Pool, cpu_count
from sklearn.metrics import f1_score, precision_score, recall_score


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


num_cores = cpu_count()


class RulesModel:

    def __init__(self, ohe_df, rules, y_dev, pos_label, neg_label):
        self.ohe_df = ohe_df
        self.oh_np_arr = np.array(ohe_df)

        self.ind_neg = list(ohe_df.columns).index(neg_label)
        self.ind_pos = list(ohe_df.columns).index(pos_label)
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.rules = rules
        self.alpha = 0.1
        self.beta = len(rules)
        self.prb_pos = sum(y_dev) / len(y_dev)
        self.prb_neg = 1 - self.prb_pos
        self.X = None

    def compute_scores(self, indx):

        positive_score = self.prb_pos
        negative_score = self.prb_neg

        applicability = 0
        positives_count = 0
        negatives_count = 0

        for _, rule in self.rules.iterrows():
            label = rule['label']
            itemset = rule['itemset']
            conditions = ' and '.join(itemset)
            if len(self.X.iloc[[indx]].query(conditions)) > 0:
                applicability += 1
                if label == self.pos_label:
                    positives_count += 1
                    positive_score *= ((rule['support'] + self.alpha) / (self.prb_pos + self.alpha * self.beta))

                    neg_count = sum(self.oh_np_arr[:, self.ind_neg])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.neg_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (neg_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        negative_score *= opp_class_score
                    else:
                        negative_score *= (self.alpha / (neg_count + self.alpha * self.beta))

                else:
                    negatives_count += 1
                    negative_score *= ((rule['support'] + self.alpha) / (self.prb_neg + self.alpha * self.beta))

                    pos_count = sum(self.oh_np_arr[:, self.ind_pos])
                    features = [list(self.ohe_df.columns).index(item) for item in list(itemset)]
                    features_support = list(np.sum(np.array(self.ohe_df[self.ohe_df[self.pos_label] == 1]
                                                            )[:, features], axis=1)).count(len(itemset))

                    opp_class_score = (features_support + self.alpha) / (pos_count + self.alpha * self.beta)
                    if opp_class_score > 0:
                        positive_score *= opp_class_score
                    else:
                        positive_score *= (self.alpha / (pos_count + self.alpha * self.beta))

        return (positive_score, negative_score, applicability, positives_count, negatives_count)

    def predict(self, data, y, alpha=20, beta=1, decision_thr=0.5):
        found_sol = []
        probas = []

        self.alpha = alpha
        self.beta = beta
        self.X = data

        pool = Pool(num_cores)
        scores = list(tqdm(pool.imap(self.compute_scores, list(range(len(data)))),
                           total=len(data), file=sys.stdout, position=0, leave=True))

        app_pos_dict = {}
        app_neg_dict = {}
        sum_dict = {}

        for score_tup in scores:
            positive_score, negative_score, applicability, positives_count, negatives_count = score_tup
            if negative_score == 0 and positive_score == 0: print('not applicable')
            if negative_score == 0 and positive_score == 0:
                found_sol.append(0)
                probas.append(0)
            elif negative_score == 0:
                found_sol.append(1)
                probas.append(1)
            elif sigmoid(math.log(positive_score / negative_score)) > decision_thr:
                found_sol.append(1)
                probas.append(sigmoid(math.log(positive_score / negative_score)))
            else:
                found_sol.append(0)
                probas.append(sigmoid(math.log(positive_score / negative_score)))

            if (positives_count in app_pos_dict):
                app_pos_dict[positives_count] += 1
            else:
                app_pos_dict[positives_count] = 1

            if (negatives_count in app_neg_dict):
                app_neg_dict[negatives_count] += 1
            else:
                app_neg_dict[negatives_count] = 1

            if (applicability in sum_dict):
                sum_dict[applicability] += 1
            else:
                sum_dict[applicability] = 1

        rules_acc = recall_score(y, found_sol, average='micro')
        print(f'Acc: {rules_acc}')

        rules_acc = recall_score(y, found_sol, average='macro')
        print(f'macro rules recall: {rules_acc}')

        rules_prec = precision_score(y, found_sol, average='macro')
        print(f'macro rules prec: {rules_prec}')

        rules_f1 = f1_score(y, found_sol, average='macro')
        print(f'macro rules f1_score: {rules_f1}')

        print(sum(y), sum(found_sol))

        self.plot(y, probas)
        self.draw_hist(app_pos_dict, app_neg_dict, sum_dict)
        print(f'{len(self.rules)} rules generated')

        return found_sol

    def predict_proba(self, data):
        found_sol = []

        pool = Pool(num_cores)
        scores = list(tqdm(pool.imap(self.compute_scores, list(range(len(data)))),
                           total=len(data), file=sys.stdout, position=0, leave=True))

        for score_tup in scores:
            positive_score, negative_score, _, _, _ = score_tup
            if negative_score == 0 and positive_score == 0: print('not applicable')
            if negative_score == 0 and positive_score == 0:
                found_sol.append(0)
            elif negative_score == 0:
                found_sol.append(1)
            else:
                found_sol.append(sigmoid(math.log(positive_score / negative_score)))
        return found_sol

    def plot(self, y, found_sol):
        import sklearn.metrics as metrics
        import matplotlib.pyplot as plt
        fpr, tpr, thresholds = metrics.roc_curve(y, found_sol)
        auc = metrics.roc_auc_score(y, found_sol)
        plt.plot(fpr, tpr, label="RulesClassification(AUC={:.2f})".format(auc))
        plt.legend(loc=4)
        plt.show()

    def plot_auc(self, data, y, alpha=None, beta=None):

        if alpha != None and beta != None:
            self.alpha = alpha
            self.beta = beta
        found_sol = self.predict_proba(data)
        self.plot(y, found_sol)

    def draw_hist(self, app_pos_dict, app_neg_dict, sum_dict, axis='x'):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        if axis == 'x':
            gs = gridspec.GridSpec(1, 3)
            x1, y1, x2, y2, x3, y3 = 0, 0, 0, 1, 0, 2
            fig = plt.figure(figsize=(15, 3))
        else:
            gs = gridspec.GridSpec(5, 1)
            x1, y1, x2, y2, x3, y3 = 0, 0, 2, 0, 4, 0
            fig = plt.figure(figsize=(11, 11))

        ax1 = fig.add_subplot(gs[x1, y1])
        ax1.bar(app_pos_dict.keys(), app_pos_dict.values(), width=0.5, color='mediumslateblue')
        ax1.set_title("Applicable Rules Histogram (Class 1)")
        ax1.set_xlabel('Count of Applicable Rules')
        ax1.set_ylabel('Number of Examples')

        ax2 = fig.add_subplot(gs[x2, y2])
        ax2.bar(app_neg_dict.keys(), app_neg_dict.values(), width=0.5, color='dodgerblue')
        ax2.set_title("Applicable Rules Histogram (Class 2)")
        ax2.set_xlabel('Count of Applicable Rules')
        ax2.set_ylabel('Number of Examples')

        ax3 = fig.add_subplot(gs[x3, y3])
        ax3.bar(sum_dict.keys(), sum_dict.values(), width=0.5, color='darkgreen')
        ax3.set_title("All Applicable Rules Histogram")
        ax3.set_xlabel('Count of Applicable Rules')
        ax3.set_ylabel('Number of Examples')

        if 0 in sum_dict:
            print(f'coverage: {1 - sum_dict[0] / len(self.X)}')
        else:
            print('coverage: 1.00')
