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
        #self.counter = 0
        self.current_score = 0
        self.duration = 0
        self.state = 0      # 0 is normal, 1 is abnormal
        temp = np.random.rand(self.order)
        temp /= np.linalg.norm(temp)
        self.x = temp

    def fit(self, X, y=None):
        states = []
        cnt = 0
        for i in X:
            print("iiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiiii = ", cnt)
            self.predict_proba(i, y)
            # Check to see if score is above threshold, if so, anomally has occured
            if self.current_score >= self.threshold:
                self.state=1 
            else:
                self.state=0
            states.append(self.state)
            cnt += 1
        return np.array(states)  # returns array of either 0 or 1 / normal or abnormal
    
    def predict(self, X, y=None):
        states = []
        ct = 0
        for j in X:
            print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjj = ",ct)
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
        print("ccccccccccccccccccccccccccccccccccccc in _proba")
        self.current_score, self.x = SingularSpectrumTransformation(win_length=self.win_length,
            n_components=self.n_components,order=self.order,lag=self.lag).score_offline(X)
        return self.current_score # returns the score

""" # Old Way
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
        print("ccccccccccccccccccccccccccccccccccccc ",self.counter)
        if self.counter == 0:
            self.current_score, self.x = start_SST(startdata=X,win_length=self.win_length,
                n_component=self.n_components,order=self.order,lag=self.lag)
            self.counter = self.counter + 1
            print("CCCCCCCCCCCCCCCCCCCCC counter in proba is ",self.counter)
        else:
            self.current_score, self.x = stream_SST(stream=X,win_length=self.win_length,
                n_component=self.n_components,order=self.order,lag=self.lag) # removed x0
        return self.current_score # returns the score
#"""