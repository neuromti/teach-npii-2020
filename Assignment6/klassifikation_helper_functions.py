import mne
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors




def load_data():
    '''
    Wir benutzen wieder die Funktion aus Übung 2 um EEG Daten mit MNE zu laden
    '''
    
    fld = Path(__file__).parent.parent / "daten"

    # Import the BrainVision data into an MNE Raw object
    raw = mne.io.read_raw_brainvision(fld / "00_rest_pre.vhdr", preload=True)

    # Read in the event information as MNE annotations
    annot = mne.read_annotations(fld / "00_rest_pre.vmrk")

    # Add the annotations to our raw object so we can use them with the data
    raw.set_annotations(annot)

    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
    event_labels = {v: k for k, v in event_ids.items()}
    for event in events:
        print("Sample {0:<10} with {1}".format(event[0], event_labels[event[-1]]))
    data = raw.get_data()
    channel_labels = raw.ch_names
    print(channel_labels)
    print(data.shape)

    return raw 



# #############################################################################
# Plot functions
def plot_model_fit(lda, X, y, y_pred):
    # #############################################################################
    # Colormap
    cmap = colors.LinearSegmentedColormap(
        'red_blue_classes',
        {'red': [(0, 1, 1), (1, 0.7, 0.7)],
         'green': [(0, 0.7, 0.7), (1, 0.7, 0.7)],
         'blue': [(0, 0.7, 0.7), (1, 1, 1)]})
    plt.cm.register_cmap(cmap=cmap)
    
    
    splot = plt.subplots()
    plt.title('Linear Discriminant Analysis')

    tp = (y == y_pred)  # True Positive
    tp0, tp1 = tp[y == 0], tp[y == 1]
    X0, X1 = X[y == 0], X[y == 1]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]

    # class 0: dots
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker='.', color='red')
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker='x',
                s=20, color='#990000')  # dark red

    # class 1: dots
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker='.', color='blue')
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker='x',
                s=20, color='#000099')  # dark blue

    # class 0 and 1 : areas
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx),
                         np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap='red_blue_classes',
                   norm=colors.Normalize(0., 1.), zorder=0)
    plt.contour(xx, yy, Z, [0.5], linewidths=2., colors='white')

    # means
    plt.plot(lda.means_[0][0], lda.means_[0][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')
    plt.plot(lda.means_[1][0], lda.means_[1][1],
             '*', color='yellow', markersize=15, markeredgecolor='grey')

    return splot


def plot_knn(knn, X, y_pred):

    from matplotlib.colors import ListedColormap
    cmap_light = ListedColormap(['red', 'blue'])
    cmap_bold = ListedColormap(['darkred', 'darkblue'])
    
    plt.figure()
    
    
    x_min, x_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.pcolormesh(xx, yy, Z, cmap = cmap_light, alpha = .3)
    
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
