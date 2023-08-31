from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from anomaly_detection.fastsst.sst import *

class SstAnomalyDetector(BaseEstimator, ClassifierMixin):

    def __init__(self,  win_length, threshold, n_components=5, order=None, lag=None,
                 is_scaled=False, use_lanczos=True, rank_lanczos=None, eps=1e-3, **kwargs):
        self.kwargs = kwargs
        # grid search attributes
        self.threshold = threshold
        self.eps = eps
        self.is_scaled = is_scaled
        self.n_components = n_components
        self.lag = lag
        if order == None:
            self.order = win_length
        else:
            self.order = order
        self.win_length = win_length
        self.use_lanczos = use_lanczos
        self.rank_lanczos = rank_lanczos        
        #self.pre_len = self.lag + self.order + self.win_length
        # internal attributes 
        self.counter = 0
        self.current_score = 0
        self.x = np.empty(1)
        self.duration = 0
        self.state = 0      # 0 is normal, 1 is abnormal

    def fit(self, X, y=None):
        states = []
        for i in X:
            self.predict_proba(i, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
        return states  # returns array of either 0 or 1 / normal or abnormal
    
    def predict(self, X, y=None):
        states = []
        for j in X:
            self.predict_proba(j, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
        return states  # returns array of either 0 or 1 / normal or abnormal
    
    def predict_proba(self, X, y=None):
        if self.counter == 0:
            self.current_score, self.x = start_SST(startdata=X,win_length=self.win_length,
                n_component=self.n_components,order=self.order,lag=self.lag)
            self.counter += 1
        else:
            self.current_score, self.x = stream_SST(stream=X,win_length=self.win_length,
                n_component=self.n_components,order=self.order,lag=self.lag) # removed x0
        return self.current_score # returns the score