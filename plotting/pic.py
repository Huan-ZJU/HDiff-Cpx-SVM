import numpy as np
import joblib
from Plot import plot


def report(results, n_top):
    table = [[0 for i in range(4)] for j in range(n_top)]
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            table[candidate][0] = results['params'][candidate].get('C')
            table[candidate][1] = results['params'][candidate].get('gamma')
            table[candidate][2] = results['mean_test_score'][candidate]
            table[candidate][3] = results['std_test_score'][candidate] * results['std_test_score'][candidate]
    return table


svc_model = joblib.load('./svc_model.pkl')

# score/variance/C/gamma
table = report(svc_model.cv_results_, 100 * 100)

plot(table, 100, 100, )
