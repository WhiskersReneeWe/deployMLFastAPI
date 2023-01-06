import pytest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from starter.ml.model import train_model, compute_model_metrics, inference


@pytest.fixture
def data_fixture():
    data = pd.read_csv("/Users/reneeliu/Desktop/mlops/deployMLFastAPI/starter/data/census.csv")
    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)
    return train, test

# Add code to load in the data.
#data = pd.read_csv("/Users/reneeliu/Desktop/mlops/deployMLFastAPI/starter/data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
#train, test = train_test_split(data, test_size=0.20)

def test_train_model(data_fixture):
    random_state = 1234
    training, testing = data_fixture
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        training, testing, test_size=0.2, random_state=10
    )
    model = LogisticRegression(random_state=random_state).fit(training_data, training_labels)

    # model = RandomForestClassifier(random_state)
    # model.fit(X_train, y_train)
    assert model is not None

def test_compute_metrics(data_fixture):
    """
    This functions tests the computation of the metrics
    """
    random_state = 1234
    training, testing = data_fixture
    training_data, testing_data, training_labels, testing_labels = train_test_split(
        training, testing, test_size=0.2, random_state=10
    )
    model = train_model(training_data, training_labels, random_state=random_state)
    predictions = model.predict(testing_data)
    precision, recall, fbeta = compute_model_metrics(testing_labels, predictions)
    assert precision >= 0.70
    assert recall >= 0.52
    assert fbeta >= 0.6

def test_inference(data_fixture):
    training, testing = data_fixture
    random_state = 1234
    training_data, testing_data, training_labels, _ = train_test_split(
        training, testing, test_size=0.2, random_state=random_state
    )
    model = train_model(training_data, training_labels, random_state=random_state)
    assert len(inference(model, testing_data)) == testing_data.shape[0]