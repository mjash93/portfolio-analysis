# libaries to help with model creation and training / optimization
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

def train_test_split(df, test_size):
	"""
		train_test_split splits the data set according to a specified percentage. Differs
		From sklearn's similar function in that order is preserved, crucial for time series
		analysis as individual observations are not independant.

		Inputs:
			df -- data frame / series consisting of experiment population
			test_size -- percentage of data set to use as testing, as such, values in (0,1)
		Returns:
			training and testing data set
	"""
	test_length = int(df.shape[0]*test_size)
	return df[:-test_length], df[-test_length:]

def rmse(y, yhat):
	"""
		returns the root mean squared error of a prediction to measure model accuracy
	"""
	return np.sqrt(mean_squared_error(y, yhat))

def lstm_dataset(data, timestep=1,n_features=1):
	"""
		LSTM models expect input to be in the following format: [samples, timesteps, features]
		where samples refer to sequences of time steps, timesteps measure the number of singular
		observations in one sequence, and features measures the amount of observations at 1 time step

		Our data is not originally formatted for this, so we must recast.

		Inputs:
			data -- data to recast to 3D.
			timestep -- number of observations per sequence, defualt is 1
			n_features -- number of observations per timestep, default is 1
		Returns:
			arrays for training and testing your model w/ specified dimensionality
	"""
	data_x, data_y = [],[]
	for i in range(data.shape[0]-timestep):
		data_x.append(data[i:i+timestep,0])
		data_y.append(data[i+timestep,0])
	data_x, data_y = np.asarray(data_x), np.asarray(data_y)
	data_x = data_x.reshape((data_x.shape[0],data_x.shape[1],n_features))
	return data_x, data_y

def fit_model(data, param):
	"""
		creates and fits a single layer LSTM model

		Inputs:
			data -- training and testing data
			param -- parameters for the model to be passed in

		Returns:
			fitted recurrent neural net build on a single LSTM layer
	"""
	inputs, nodes, epochs, batch_size, features = param
	train, test = data

	train_x, train_y = lstm_dataset(train, timestep=inputs,n_features=features)
	test_x, test_y = lstm_dataset(test, timestep=inputs,n_features=features)

	rnn = Sequential()
	rnn.add(LSTM(nodes,input_shape=(inputs, features)))
	rnn.add(Dense(1))
	rnn.compile(optimizer='adam',loss='mean_squared_error')
	history = rnn.fit(train_x, train_y, epochs = epochs, batch_size=batch_size,
								verbose=0, shuffle=False, validation_data=(test_x,test_y))
	return rnn, history

def walk_through_validation(data, param):
	"""
		Unlike other ML models, time series forecasting cannot use standard CV methods as they
		assume that the observations are independant, which is not the case for time series
		analysis. walk_through_validation fits and forecasts and returns the root mean squared
		error of the predicutions

		Input:
			data -- data to be unpacked
			param -- parameteres to be passed onto other functions

		Returns:
			error -- root mean squared error of prediction
	"""
	# unpack data
	train, test, scale = data
	inputs,_,_,_, features = param
	# scale data to range (0,1)
	train_scaled, test_scaled = scale.fit_transform(train.values.reshape(-1,1)), scale.fit_transform(test.values.reshape(-1,1))
	data_scaled = train_scaled, test_scaled
	# fit our model
	model, history = fit_model(data_scaled, param)
	test_x, test_y = lstm_dataset(test_scaled, timestep=inputs,n_features=features)
	prediction = model.predict(test_x)
	test_y = scale.inverse_transform(test_y.reshape(-1,1))
	# rescale predictions back to standard metric
	prediction = scale.inverse_transform(prediction)
	error = rmse(test_y, prediction)
	return error

def model_evals(data, param, key_dict,evals=1):
	""""
		We run tests with each set of parameters a fixed amount of time and take the average
		score to ensure better results as the weights in our model are randomly initialized.
		Inputs:
			data -- data to be unpacked for additional functins
			param -- parameters to be evaluated
			key_dict -- dictionary with hyperparameter values
			evals -- number of iterations to run over a single parameter
		Returns:
			key -- string of the parameters used for specific evaluation
			score -- average score across all iterations
	"""
	key = str(param)

	scores = [walk_through_validation(data, param) for _ in range(evals)]
	key_dict[key] = param
	print("Hyperparams {} evaluated!".format(param))
	return key, param, np.mean(scores)

def grid_search(data, params):
	"""
		grid search is used to find the optimal batch of parameters for our model. We make our
		own funciton instead of a sklearn one for the same reason we recreated train_test_split

		Input:
			data -- data to be unpacked
			params -- list of all combinations of parameters to be tested
		Returns:
			results of all parameters alongside their corresponding error value
	"""
	key_dict = {}
	results = [model_evals(data, param, key_dict) for param in params] # iterates through all parameter combos
	results.sort(key=lambda x: x[2]) # sorts list based off of rmse

	return results

def parameters(inputs, nodes, epochs, batch_size, features):
	"""
		Creates list of all possible combinatinos of parameters

		Inputs:
			parameters -- all stated parameters for the analysis
		Returns:
			params -- list of all combinations of parameters
	"""
	params = []
	for a in inputs:
		for b in nodes:
			for c in epochs:
				for d in batch_size:
					for e in features:
						params.append([a,b,c,d,e])
	return params

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	"""
	Frame a time series as a supervised learned dataset.
	Inputs:
		data -- sequence of observations as a list or numpy array.
		n_in -- number of lag observations as input (x)
		n_out -- number of observations as output (y)
		dropnan -- boolean whether or not to drop rows with nan values
	 Returns:
	 	Pandas DataFrame of series framed for supervised learning.
	"""
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = [], []
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var{}(t-{})'.format(j+1, i)) for j in range(n_vars)]
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var{}(t)'.format(j+1)) for j in range(n_vars)]
		else:
			names += [('var{}(t+{})'.format(j+1, i)) for j in range(n_vars)]
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	if dropnan:
		agg.dropna(inplace=True)
	return agg
