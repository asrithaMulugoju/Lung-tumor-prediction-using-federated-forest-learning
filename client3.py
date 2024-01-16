from utils import load_data,get_params,set_params
import numpy as np
import flwr as fl
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.simplefilter('ignore')



# Create the flower client
class FlowerClient(fl.client.NumPyClient):

    def get_parameters(self, config):
        return get_params(model)

    
    def fit(self, parameters, config):
        set_params(model, parameters)
        model.fit(X_train, y_train)
        trained_params = get_params(model)

        return trained_params, len(X_train), {}

    
    def evaluate(self, parameters, config):
        set_params(model, parameters)
        y_pred = model.predict(X_test)
        loss = log_loss(y_test, y_pred, labels=[0, 1])
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        return loss, len(X_test), {"Accuracy": accuracy, "Precision": precision, "Recall": recall, "F1_Score": f1}

if __name__ == "__main__":
    client_id = 3
    X_train, X_test, y_train, y_test = load_data(file_path="survey_lung_cancer.csv",client_id=client_id)

    model = RandomForestClassifier(
        class_weight='balanced',
        criterion='entropy',
        n_estimators=100,
        max_depth=40,
        min_samples_split=2,
        min_samples_leaf=1,
    )

    model.fit(X_train, y_train)

    fl.common.logger.configure(identifier="client_logs", filename="clien3.txt")

    fl.client.start_numpy_client(server_address="localhost:5040", client=FlowerClient())