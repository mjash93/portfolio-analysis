# libraries to help w/ data extraction
import requests
import requests_cache
import lxml.html as lx
import json
import time

# libraries to help w/ data analysis & manipulation
import numpy as np
import pandas as pd
import scipy.optimize as spo

def read_key(keyfile):
	"""
		reads the content of a file (api key) into a variable to use for further usage
	"""
	with open(keyfile) as f:
		return f.readline().strip("\n")


def get_data(key, tickers, url='https://www.alphavantage.co/query?', outputsize='full'):
	"""
		url and output size are fixed defaults that can be changed though refer to the documentation
		for the correct vernacular. Returns a dict where the keys are the stock symbols
		queried and the values are dicts where each key is a date w/ corresponding daily statistics
	"""
	data = {}
	for tick in tickers:
		response = requests.get(url, params={
			'function': 'TIME_SERIES_DAILY_ADJUSTED',
			'apikey': key,
			'symbol': tick,
			'outputsize': outputsize,
			})
		if response.status_code > 299: 
			raise ConnectionError('Trouble connecting to alphavantage')
		if len(list(response.json().keys())) is not 2:
			raise IndexError('TICK:{}\nMESSAGE:{}'.format(tick,response.json()['Note']))
		tsd = list(response.json().keys())[1] # time series daily
		data[tick] = response.json()[tsd]
		time.sleep(15)
	return data

def get_rfr(url='https://www.federalreserve.gov/releases/h15/'):
	"""
		function scrapes the risk free rate from the federal reserve and returns it as a
		floating point value.
	"""
	response = requests.get(url)
	if response.status_code > 299:
		raise ConnectionError('Trouble connecting to federalresere.gov')
	html = lx.fromstring(response.text)
	rfr = float(html.xpath("//div[@class='data-table']//td")[4].text_content().strip())
	return rfr

def create_df(data, date_start=None, metric='adj close',norm=False):
	"""
		create's a daraframe where the columns represent each stock's chosen metric. default is
		adjusted close. You can input a start date if you want to take slices smaller than
		the range of the data.
	"""
	metric_dict = {'open':'1. open', 'high': '2. high','low': '3. low',
					'close': '4. close','adj close': '5. adjusted close','volume':
					'6. volume','dividend': '7. dividend amount','split':'8. split coefficient'}
	tickers = list(data.keys())
	if date_start is None:
		date_start = list(data[tickers[0]].keys())[-1]
	date_end = list(data[tickers[0]].keys())[0]

	date_range = pd.date_range(date_start, date_end)
	df = pd.DataFrame(index=date_range)

	for tick in tickers:
		dft = pd.DataFrame(data[tick]).T.rename({metric_dict[metric]: tick}, axis=1).astype(float)
		dft = dft.loc[:, tick].reindex(index=dft.index[::-1]).astype(float)
		df = df.merge(dft, left_index=True, right_index=True, how='left')
		df.index.name = 'date'
	df.fillna(method='bfill',inplace=True)
	if norm:
		df = df / df.iloc[0,:]
	return df

def sharpe(w0, df):
	if type(w0) is not np.ndarray:
		w0 = np.asarray(w0)

	avg_ann_ret = df.pct_change()[1:].mean() *252
	ann_cov = df.pct_change().cov() * 252 # daily returns cov matrix

	norm_w = np.asarray([x / np.sum(w0) for x in w0])
	returns = np.dot(norm_w, avg_ann_ret)
	volatility = np.sqrt(np.dot(norm_w.T, np.dot(ann_cov, norm_w)))
	sharpe = returns / volatility
	return -1 * sharpe

def constraints(w0):
	return np.sum(w0) - 1

def optimize_weights(w0, df):
	#bnd = ((0.0,1.0) for x in range(len(w0)))
	bnd = tuple([(0.0,1.0) for x in range(len(w0))])
	cons = {'type':'eq', 'fun': constraints}
	optimizer = spo.minimize(sharpe,w0,args=(df),method='SLSQP',
							bounds=bnd,tol=1e-05,constraints=cons)
	return optimizer.x

def bollinger(df):
	N = df.shape[1]
	df = df / df.iloc[0,:]
	for tick in df.columns:
		df[tick+' 30 day ma'] = df[tick].rolling(window=20).mean()
		df[tick+' 30 day std'] = df[tick].rolling(window=20).std()
		df[tick+' upper band'] = df[tick+' 30 day ma'] + df[tick+' 30 day std']*2
		df[tick+ ' lower band'] = df[tick+' 30 day ma'] - df[tick+' 30 day std']*2

	return df

def EF(df, num_assets, num_port):
	np.random.seed(1)
	port_returns, port_volatility, stock_sharpe, stock_weights = [], [], [], []
	rfr = get_rfr() * 10**(-2) # scrape the risk free rate from the FED

	adr = df.pct_change()[1:].mean() # average daily returns
	aar = 252 * adr # average annual returns (252 trading days in a year)

	dcov = df.pct_change().cov() # daily returns cov matrix
	acov = dcov * 252 # annual returns , cov matrix

	for p in range(num_port):
		weights = np.random.random(num_assets)
		weights /= np.sum(weights)
		returns = np.dot(weights, aar)
		volatility = np.sqrt(np.dot(weights.T, np.dot(acov, weights)))
		sharpe = returns / volatility

		port_returns.append(returns)
		port_volatility.append(volatility)
		stock_sharpe.append(sharpe)
		stock_weights.append(weights)

	portfolio_dict = {
		'returns': port_returns, 'volatility': port_volatility,
		'sharpe ratio': stock_sharpe,
		}

	for counter, tick in enumerate(list(df.columns)):
		portfolio_dict[tick +' Weight'] = [weight[counter] for weight in stock_weights]

	dfn = pd.DataFrame(portfolio_dict)
	return dfn

def create_portfolio(df, inv, weight):
	"""
		create's a portfolio along with summary statistics. returns both the portfolio and
		summary statistics in a dict.
	"""
	portfolio_statistics = {}
	opt_weight = optimize_weights(weight, df)
	portfolio = round(((df/df.iloc[0,:])*inv*opt_weight).sum(axis=1), 3)

	daily_returns = portfolio.pct_change()[1:]
	rfr = (1+get_rfr()/100)**(1/63) - 1

	portfolio_statistics['cumulative returns'] = round(portfolio[-1] / portfolio[0]-1, 3)
	portfolio_statistics['average daily returns'] = round(daily_returns.mean(),5)
	portfolio_statistics['average daily standard dev'] = round(daily_returns.std(),5)
	portfolio_statistics['sharpe ratio'] = round((np.sqrt(252)*(daily_returns - rfr).mean()/daily_returns.std()),3)
	portfolio_statistics['allocation'] = opt_weight

	return portfolio, portfolio_statistics
