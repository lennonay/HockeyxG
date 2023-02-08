import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

#adopted from https://github.com/HarryShomer/xG-Model
def get_roc(actual, predictions):
    """
    Get the roc curve (and auc score) for the different models
    """
    fig = plt.figure()
    plt.title('ROC Curves')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    colors = ['b', 'g', 'p']

    for model, color in zip(predictions.keys(), colors):
        # Convert preds to just prob of goal
        preds = [pred[1] for pred in predictions[model]]

        false_positive_rate, true_positive_rate, thresholds = roc_curve(actual, preds)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.plot(false_positive_rate, true_positive_rate, label=' '.join([model + ':', str(round(roc_auc, 3))]))

    # Add "Random" score
    plt.plot([0, 1], [0, 1], 'r--', label=' '.join(["Random:", str(.5)]))

    plt.legend(title='AUC Score', loc=4)
    fig.savefig("results/ROC_xG.png")

if __name__ == "__main__":

    train_df = pd.read_csv('data/pre_processed/train_df.csv')
    test_df = pd.read_csv('data/pre_processed/test_df.csv')

    X_train = train_df.drop('goal', axis = 1)
    y_train = train_df['goal']

    X_test = test_df.drop('goal', axis = 1)
    y_test = test_df['goal']

    lr_file = 'src/model/pipe_lr.pkl'
    gbc_file = 'src/model/pipe_gbc.pkl'
    rf_file = 'src/model/pipe_rf.pkl'

    # load the model from disk
    pipe_lr = pickle.load(open(lr_file, 'rb'))
    pipe_gbc = pickle.load(open(gbc_file, 'rb'))
    pipe_rf = pickle.load(open(rf_file, 'rb'))

    preds = {
        "Random Forest": pipe_rf.predict_proba(X_test),
        "Gradient Boosting": pipe_gbc.predict_proba(X_test),
        "Logistic Regression": pipe_lr.predict_proba(X_test)}

    get_roc(y_test, preds)