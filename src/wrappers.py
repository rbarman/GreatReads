from surprise import Dataset
from surprise import Reader
from surprise.model_selection import train_test_split
from surprise import accuracy

## Some Suprise wrappers

def get_train_test_split(ratings_df, rating_scale, column_names,test_size = .2):
  '''
  Create a Suprise dataset from Pandas Dataframw with user/item ratings and create train/test split

  Params
    - ratings_df = Pandas dataframe with user and item rating interactions 
    - rating_scale = tuple with scale of ratings -> (0,5)
    - column_names = column names that correspond to user id, item id, and rating IN THAT ORDER -> ['user_id', 'book_id', 'rating']
    - test_size = size of test set as a percentage of rating_df's size
  
  Returns
    - trainset = Training partition (Suprise Trainset)
    - testset = Testing partition (list of userid,itemid,rating tuples)

  '''

  # create a reader with the specified rating scale
  reader = Reader(rating_scale=rating_scale)
  # Create a suprise Dataset
  data = Dataset.load_from_df(ratings_df[column_names], reader)
  # split into train and test
  trainset, testset = train_test_split(data, test_size=test_size)

  return trainset, testset

def get_metrics(predictions):
  '''
  Compute accuracy metrics

  Params
    - predictions = list of Suprise Predictions
  Returns
    - Dictionary with metrics

  TODO: Review https://hkh12.scripts.mit.edu/mlgp/Weeks/week15/evaluationRecoEngine.pdf for more relevant metrics

  '''
  metric_dict = {}
  metric_dict['RMSE'] = accuracy.rmse(predictions, verbose=False)
  # TODO: fcp takes long time + is it relevant?
  #metric_dict['FCP'] = accuracy.fcp(predictions, verbose=False)
  metric_dict['MAE'] = accuracy.mae(predictions, verbose=False)
  metric_dict['MSE'] = accuracy.mse(predictions, verbose=False)

  return metric_dict

def train_and_evaluate(train_data, test_data, SurpriseAlgo):
  '''Train Suprise algorithm with default hyperparams on the train data and evaluate on test set 

    params:
      train_data = train partition from get_train_test_split()
      test_data = test partition from get_train_test_split()
      SupriseAlgo = specific algorithm class (https://surprise.readthedocs.io/en/stable/prediction_algorithms_package.html)
    
    returns:
      predictions = list of Suprise Predictions
      metrics = dictionary of metrics after applying trained model to test set

    TODO: incorporate hyper params
  '''

  algo = SurpriseAlgo()
  print(algo)
  algo.fit(train_data)
  predictions = algo.test(test_data)
  metrics = get_metrics(predictions)
  print(metrics)

  return predictions, metrics