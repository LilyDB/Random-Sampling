
import math
import numpy as np
from scipy import stats
from scipy.stats import norm
import pandas as pd
import seaborn as sns
from math import ceil
from sklearn.model_selection import StratifiedShuffleSplit

# COMMAND ----------

class sampling:
    """
    Input:
    N: population size
    e: margin of error, desired level of precision
    cl: desired confidence level with 2 decimal precison
    p: proportion of the population which has the attribute in question
        (i.e: 30% of vessels lacking running hour)

    Output:
    Sample size: minimum sample size needed to estimate the true population proportion 
                    with the required margin of error and confidence level.
    """
    
    def __init__(self,N,e,cl,p,inf):
        self.N = N
        self.e = e
        self.cl = cl
        self.p= p
        self.inf = inf
    
    def cochran(self):
        """
        Use when proportion of the population which has the attribute in question is known
        cochran_n = (Z**2 * p * q) /e**2
        """

        # generate confidence_level, 2 decimal
        cl_list = [round(i,2) for i in  np.arange(0,1,0.01).tolist()]

        # list comprehension
        alpha_list =  [1 - i for i in cl_list]
        zscore_list = [norm.ppf(1-(i/2)) for i in alpha_list]

        # dict comprehension
        dict = {zscore_list[i]:cl_list[i] for i in range(len(cl_list))}
        
        # if input is valid
        if (0<=self.N \
            #error should <=10%
            and 0<=self.e<=0.1
            # confidence level should >= 80%
            and 0.8<=self.cl<=1\
            and 0<=self.p<=1):
            
            # Loop in the dict to find key
            for zscore, value in dict.items():
                if self.cl==value:
                    numerator = zscore**2*(self.p*(1-self.p))
                    denom = self.e**2
                    if self.inf == True:
                        #inifinte sample size
                        s_0 = ceil(numerator/denom) 
                        return s_0
                    else:
                        s_0 = numerator/denom
                        s = ceil(self.N*s_0/(self.N+s_0-1))
                        return s
        else:
            raise ValueError('Parameters are not valid')


    def slovin(self):
        """
        Use when only population size is known
        """
        s = ceil(self.N/(1+self.N*self.e**2))
        return s


    def stratified_sample(self,df,strata,size=None):  
           """
           input
           df: dataframe for sampling including variables for stratify 
           strata: variable (or combined variable) for stratify
           size: desired minimum sample size
           
           output:
           report on number of items for each stratum
           list of item id was stratified chosen
           """
        test_size = size/len(df)
        df['num_item'] = df.index
        df_s = df[['num_item','strata']]
        
        sss = StratifiedShuffleSplit(n_splits=10, # by default
                                     test_size=test_size, 
                                     random_state=0)
           
        for x, y in sss.split(df_s, df_s['strata']):
            stratified_random_sample = df_s.iloc[y].sort_values(by='strata')
            s = stratified_random_sample.groupby('strata').count()
            print(s)
            return stratified_random_sample


# COMMAND ----------

s = sampling(len(df),0.1,0.95,0.3,inf=False)
s.stratified_sample(df,df['strata'],size=s.slovin())
