'''
This is a record book of how to implement a sklearn transformer, and/or your own Pipeline and FeatureUnion

sklearn nicely provides the base classes BaseEstimator and a mixin TransformerMixin for this purpose. BaseEstimator provides two methods get_params() and set_params(). Those are the methods used when doing hyperparameter search. Let's take a look at these methods first
'''

@classmethod
def _get_param_names(cls):
    init = getattr(cls.__init__, 'deprecated_original', cls.__init__)
    if init is object.__init__:
        return []
    
    init_signature = signature(init)
    parameters = [p for p in init_signature.parameters.values()
                    if p.name != 'self' and p.kind != p.VAR_KEYWORD]

    ## and something else
    ##

def get_params(self, deep=True):
    out = dict()
    for key in self._get_param_names():
        value = getattr(self, key, None)
        if deep and hasattr(value, 'get_params'):
            deep_items = value.get_params().items()
            out.update((key + '__' + k, val) for k, val in deep_items)
        out[key] = value 
    return out 

'''
There are few things 
1. get_params() calls a hidden class method _get_param_names() to retrieve the keyword parameters explicitly defined on class __init__ function. **kwargs is not welcomed here. If we track back signature function in .utils.fixes module, we will find out it is only a wrapper to have uniform interface for Python 2 and 3. For Python 2 it calls inspect.getargspec method and for Python 3 it calls funcsigs.signature method to retrieve the class.__init__ signature, remove self and *vars, and return the keys of kwargs in a list. This is for sklearn version 0.18. Actually for 0.16 and earlier version, only inspect.getargspec was used (it is depcrecated). (I skipped 0.17). This means, when we define our own transformers with hyperparameters, and we want to do automated hyperparameter search, we need to explicitly define those hyperparameter as keyward arguments of the __init__ function. 
2. get_params(deep=True) has deep flag default to True. When the flag is set True, it will recursively search down the parameters which are of BaseEstimators, until it reaches the level this flag is set False or to the object level.
3. When it goes deeper for parameters, it uses double underscore '__' to define the namespace for sub-parameters. The set_prams() method uses the same rule to parse the parameter name when setting the values. 
4. Also worth paying attention setattr calls class.__setattr__ to set the attributes. Try not to mess up with class.__setattr__, especially do not call setattr in class.__setattr__.   
5. We do not have to do anything with these methods. 

Now let's take a look at TransformerMixin 
'''

class TransformerMixin(object):
    def fit_transform(self, X, y=None, **fit_parameters):

        '''
        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.
        y : numpy array of shape [n_samples]
            Target values.
        '''
        if y is None:
            return self.fit(X, **fit_parameters).transform(X)
        else:
            return self.fit(X, y, **fit_parameters).transform(X)

'''
Also, there are a few things to notice
1. Those it does not define fit() and transform() as abstract methods, but implicitly it requires the implementation of these two methods.
2. The y numpy array shape instruction is not very accurate here, better to be 2D, using y.reshape(-1,1)
3. y is required. Even you override fit_transform method, a lot of other methods still assume there is a y parameter at the position. So to be compatable with the system, put y=None.
4. Notice there are fit functions require y, and others require no y, legacy issues. 


So now let's build a customized transformer that normalizes data using logistic function. Normalization can bring a lot of benefits such as unified input, data bias and variation removed, better computation efficiency, less numerical error, et.. Sigmoid function, as an alternative to min-max and standardization, combines linear and non-linear approximation. It keeps the linearity for the majority of data, while for extreme values (outliers) it tapers the values and maintains the ordinal order. 

We will also set two parameters bias and scale as a population correction when there is data shift. 

If the parameter list is too long, you can use the inspect.getargspec module to help initialization.
'''

from sklearn.base import BaseEstimator, TransformerMixin
import inspect 
import numpy as np 
import matplotlib.pyplot as plt 

def sigmo(m, sig):
    def f(x):
        return 1.0/(1.0 + np.exp(-(x-m)/sig))
    return f 


class Sig_trans(BaseEstimator, TransformerMixin):
    def __init__(self, bias=0, scale=1.0):
        ## this is an over kill for these two parameters, we can just do
        # self.bias = bias
        # self.scale = scale
        z = [p.name for p in inspect.signature(self.__init__).parameters.values()]
        l = locals()
        for k in z:
            setattr(self, k, l[k])

    def _fit(self, X, y=None):
        self.mean = X.mean()
        self.std = X.std()
        self.sigmoid = sigmo(self.mean, self.std)

    def fit(self, X, y=None):
        self._fit(X)
        return self 

    def transform(self, X):
        if not hasattr(self, 'sigmoid'):
            raise AttributeError('Fit the transformer first')
        return self.sigmoid(X)

if __name__ == '__main__':
    X_train =  np.random.normal(0, 2.0, (200, 1))
    X_test = np.random.normal(0, 5.0, (20, 1))

    T = Sig_trans()
    X_train_p = T.fit_transform(X_train)
    X_test_p = T.transform(X_test)

    fig = plt.figure()
    plt.plot(x, xp, 'r.', label='fitting data', alpha=0.3)
    plt.plot(y, yp, '.', label='transform data')

    plt.legend()
    plt.xlabel('X_train')
    plt.ylabel('X_test')
    plt.ylim(0,1)
    xlim = plt.xlim()
    plt.plot(xlim, [0.5, 0.5], '-g', linewidth=1)
    plt.plot([0,0],[0,1], '-g', linewidth=1)
    plt.xlim(xlim)
    plt.show()  


## figure added


'''
now follows an example of this transformer as part of a Pipeline object
'''



'''
transformer combiner example
'''




        





