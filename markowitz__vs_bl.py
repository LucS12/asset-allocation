#Necessary Packages:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as web
from statsmodels.stats.correlation_tools import cov_nearest

#Gather closing prices for TSLA and KO stocks:
stocks = ['TSLA', 'KO']
data = web.DataReader(stocks, start='2018-01-01', end='2021-10-27', data_source='stooq').Close

#Calculate yearly returns and covariance matrix:
yearly_ret = data.resample('Y').last().pct_change().mean()  #Average yearly returns
cov = data.pct_change().cov()

#Empty lists to store weights, returns, and volatilities of portfolios made:
weights = []
returns = []
vols = []

tesla_w = 0   #TSLA initial weight

#Loop to calculate each portfolio with 100 different weights for TSLA:
for i in range(101):
  #Weights calculation:
  coke_w = 1 - tesla_w              #Find coke's weight given TSLA's
  w = np.array([tesla_w, coke_w])   #groups the two weights as one 
  weights.append(w)                 #Adds the two into list
  tesla_w += 0.01                   #Increase tesla weight each iteration by 0.01

  #Returns:
  ret = np.dot(w, yearly_ret)       #Weighted sum between returns and weights
  returns.append(ret)               #Add it to the list

  #Volatility:
  var = np.dot(w.T, np.dot(cov, w))        #w.T*Cov*w as shown previously
  yearly_vol = np.sqrt(var)*np.sqrt(252)   #Standard deviation is volatility
  vols.append(yearly_vol)

#Putting made lists into a dictionary:
port = {'Returns': returns, 'Volatility':vols}

#Placing weights into dictionary made with a list comprehension:
for counter, symbol in enumerate(data.columns.tolist()):
  port[symbol + '_Weight'] = [weight[counter] for weight in weights]

#Make a dataframe from the dictionary buil:
port_df = pd.DataFrame(port)

# Minimum Volatility Portfolio:
min_vol = port_df.iloc[port_df.Volatility.idxmin()]

#Plot Tesla Weight vs. Volatility
plt.subplots(figsize=(10,8))
plt.title('Tesla Weight vs Port. Volatility', fontsize=20)
plt.xlabel('Tesla Weight', fontsize=15)
plt.ylabel('Volatility', fontsize=15)
plt.scatter(port_df.TSLA_Weight, port_df.Volatility, s=5)
plt.scatter(min_vol[2], min_vol[1], s=350, color = 'y', marker = '*')
plt.show()

#Market Cap. Weights:
mcs = web.get_quote_yahoo(stocks)['marketCap'].values    #Geta market caps of each as np array
mcs_w = mcs / mcs.sum()    #NumPy lets us do vectorized operation

#Risk-Aversion and Covariance Matrix (S):
A = 1.2
S = cov

#Implied Equilibrium Excess Returns (pi):
    #pi = 2A*S*w
pi = 2.0*A*np.dot(S, mcs_w)   

#Views (Q): Tesla will outperform coca cola by 10% 
Q = np.array([0.5])

#Link Matrix (P): 1 for positive viewed asset and -1 for negatively viewed asset
P = np.array([1, -1])

#Scalar (tau), c, and Uncertainty of views matrix (omega):
    #tau = 1 / length of time series 
    #c = 1 as default
    #omega = 1/c*P*S*P^T 
c = 1  
tau = 1/float(len(data))
omega = (1/c) * np.dot(np.dot(P, S), P.T) 

#BL Excess Return:
    # = pi + tau*S*P^T * (tau*P*S*P^T + omega)^-1 * (Q - P*pi)
bl_returns = pi + (np.dot(tau*np.dot(S, P.T), 
             (tau*np.dot(np.dot(P, S), P.T) + omega)**-1) * (Q - np.dot(P, pi)))

#BL Covariance Matrix:
    # = (1+tau)*S - tau^2*S*P.T * (tau*P*S*P.T + omega)^-1 * P*S
bl_S = (1+tau)*S - np.dot(np.dot(tau**2*np.dot(S, P.T), 
        (np.dot(tau*np.dot(P, S), P.T) + omega)**-1), np.dot(P, S))

sym_S = (bl_S + bl_S.T) / 2      #Make it symmetric
semidef_S = cov_nearest(sym_S)   #Make it strict positive semi-definite

#BL Posterior Weights:
    # w' = postRet * (A*postS)^-1
post_W = np.dot(bl_returns, np.linalg.inv(A*semidef_S))     
post_W = post_W / post_W.sum()   

#Printing BL Weights:
print('\nB-L Adjusted Portfolio Weights:')
print('Tesla: ' + str(round(post_W[0], 2)))
print('Coca-Cola: ' + str(round(post_W[1],2)))

#Setting weights to respective lists:
mark_ws = [min_vol[2], min_vol[3]]
bl_ws = post_W

#Placing weights into dataframe:
frame = pd.DataFrame([mark_ws, bl_ws],
                     columns = stocks,
                     index = ['Min. Vol. Weights', 'B-L Weights'])

frame.T.plot(kind='bar', figsize=(12,8))  #Plotting the dataframe as a bar chart
plt.legend(fontsize=15)                   #Increasing legend fontsize
