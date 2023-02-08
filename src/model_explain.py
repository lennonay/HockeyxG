import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import altair as alt
import vl_convert as vlc

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

def save_chart(chart, filename, scale_factor=1):
    #save_chart function reference from Joel Ostblom
    '''
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    '''
    #saves altair object as png
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

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
    lr_base_file = 'src/model/pipe_lr_base.pkl'

    # load the model from disk
    pipe_lr = pickle.load(open(lr_file, 'rb'))
    pipe_gbc = pickle.load(open(gbc_file, 'rb'))
    pipe_rf = pickle.load(open(rf_file, 'rb'))
    pipe_lr_base = pickle.load(open(lr_base_file, 'rb'))

    preds = {
        "Random Forest": pipe_rf.predict_proba(X_test),
        "Gradient Boosting": pipe_gbc.predict_proba(X_test),
        "Logistic Regression": pipe_lr.predict_proba(X_test)}

    get_roc(y_test, preds)

    categorical_features  = ['shotType', 'lastEventCategory','team', 'shooterLeftRight' ]
    binary_features = ['isPlayoffGame','shootingTeamEmptyNet', 'defendingTeamEmptyNet']

    numeric_features = ['arenaAdjustedShotDistance',
    'distanceFromLastEvent',
    'shotAngle',
    'xCordAdjusted',
    'yCordAdjusted','defendingTeamDefencemenOnIce','defendingTeamForwardsOnIce',
    'defendingTeamGoals', 'shootingTeamDefencemenOnIce','shootingTeamForwardsOnIce','shootingTeamGoals']

    ohe_features = pipe_lr_base.named_steps["columntransformer"].named_transformers_["pipeline"].get_feature_names_out(categorical_features).tolist()
    feature_names = (numeric_features + binary_features + ohe_features)

    data = {
        "coefficient": pipe_lr_base.named_steps["logisticregression"].coef_.flatten().tolist(),
        "magnitude": np.absolute(
            pipe_lr_base.named_steps["logisticregression"].coef_.flatten().tolist()
        ),
    }
    coef_df = pd.DataFrame(data, index=feature_names).sort_values(
        "magnitude", ascending=False
    ).reset_index()

    coef_df.to_csv('results/feature_importance.csv', index = False)

    chart = alt.Chart(coef_df.head(15), title = 'Feature Importance from Logistic Regression').mark_bar().encode(
    x = 'coefficient',
    y = alt.Y('index', sort = '-x'),
    color = alt.Color(scale = alt.Scale(scheme='dark2')))

    save_chart(chart, 'results/feature_importnace.png')