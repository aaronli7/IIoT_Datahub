from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, OutlierMixin
from anomaly_detection.fastsst.sst import *

class SstDetector(BaseEstimator, OutlierMixin):

    def __init__(self,  win_length, order, n_components, lag,is_scaled, use_lanczos, rank_lanczos, eps, threshold=0, **kwargs):
        self.kwargs = kwargs
        # grid search attributes
        self.threshold = threshold        
        if order == None:
            self.order = win_length
        else:
            self.order = order       
        self.eps = eps
        self.is_scaled = is_scaled
        self.n_components = n_components
        self.lag = lag
        self.win_length = win_length
        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos        
        self.current_score = 0
        self.duration = 0
        self.state = 0      # 0 is normal, 1 is abnormal
        self.x = 0

    def fit(self, X, y=None):
        state_0 = []
        state_1 = []
        cnt = 0
        if cnt == 0:
            temp = np.random.rand(self.order)
            temp /= np.linalg.norm(temp)
            self.x = temp
        for i in X:
            self.predict_proba(i, y)
            # Check to see if score is above threshold, if so, anomally has occured
            #if self.current_score >= self.threshold:
            #    self.state=1 
            #else:
            #    self.state=0
            if y[cnt] == 0:
                state_0.append(self.current_score)
            else:
                state_1.append(self.current_score)
            cnt += 1
        average_of_zeros = np.average(np.array(state_0))
        average_of_ones = np.average(np.array(state_1))
        if average_of_ones >= average_of_zeros:
            diff = abs(average_of_ones - average_of_zeros)
            threshold = average_of_zeros + diff
        else:
            diff = average_of_zeros - average_of_ones
            threshold = abs(average_of_ones + diff)
        self.threshold = threshold
        #print("states: ",states)
        #return np.array(states)  # returns array of either 0 or 1 / normal or abnormal
        return
    
    def predict(self, X, y=None):
        states = []
        ct = 0
        if ct == 0:
            temp = np.random.rand(self.order)
            temp /= np.linalg.norm(temp)
            self.x = temp
        for j in X:
            self.predict_proba(j, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
            ct += 1
        return np.array(states)  # returns array of either 0 or 1 / normal or abnormal
    
    def predict_proba(self, X, y=None):
        self.current_score, self.x = SingularSpectrumTransformation(win_length=self.win_length,x0=self.x,
            n_components=self.n_components,order=self.order,lag=self.lag,is_scaled=self.is_scaled,
            use_lanczos=self.use_lanczos,rank_lanczos=self.rank_lanczos,eps=self.eps).score_online(X)
        #print("score: ",self.current_score)
        return self.current_score # returns the score