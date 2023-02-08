import pandas as pd
import pickle
import matplotlib as plt
from sklearn.metrics import roc_curve, auc

train_df = pd.read_csv('data/pre_processed/train_df.csv')
test_df = pd.read_csv('data/pre_processed/test_df.csv')

X_train = train_df.drop('goal', axis = 1)
y_train = train_df['goal']

X_test = test_df.drop('goal', axis = 1)
y_test = test_df['goal']

lr_file = 'src/model/lr_model.pkl'
gbc_file = 'src/model/gbc_model.pkl'
rf_file = 'src/model/rf_model.pkl'

# load the model from disk
lr_model = pickle.load(open(lr_file, 'rb'))
result = lr_model.score(X_test, y_test)
print(result)


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
    fig.savefig("ROC_xG.png")

#preds = {
    #"Random Forest": pipe_rf.predict_proba(X_test),
    #"Gradient Boosting": pip_gbc.predict_proba(X_test),
    #"Logistic Regression": pipe_lr_unopt.predict_proba(X_test)}

#get_roc(y_test, preds)