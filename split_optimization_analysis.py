from rdkit import Chem
from rdkit.Chem import AllChem
import gzip
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import pairwise_distances, jaccard_similarity_score, precision_recall_curve, auc, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from ukySplit import ukyDataSet


def finger(mol):
    fprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    return list(fprint)


def make_prints(s):
    inf = gzip.open(s)
    gzsuppl = Chem.ForwardSDMolSupplier(inf)
    mols = [x for x in gzsuppl if x is not None]
    prints = [finger(mol) for mol in mols]
    prints = pd.DataFrame(prints).dropna().drop_duplicates().values
    return prints


def nn_similarity(probs, nn_preds):
    similarities = []
    for t in np.linspace(0, 1, num=100):
        preds = (probs > t).astype(bool)
        similarities.append(jaccard_similarity_score(preds, nn_preds))
    return np.max(similarities)


def PR_AUC(probs, labels, weights=None):
    if weights is None:
        precision, recall, _ = precision_recall_curve(labels, probs)
    else:
        precision, recall, _ = precision_recall_curve(labels, probs, sample_weight=weights)
    return auc(recall, precision)


def main(decoys_file, actives_file, output_file):
    active_prints = make_prints(actives_file)
    decoy_prints = make_prints(decoys_file)
    fingerprints = np.vstack([active_prints, decoy_prints]).astype(bool)
    labels = np.vstack([np.ones((len(active_prints), 1)), np.zeros((len(decoy_prints), 1))]).flatten().astype(int)
    distances = pairwise_distances(fingerprints, metric='jaccard')

    # A full splitting of the data is into test, buffer, training, and validation sets.
    # First a random split into test and temp_training, then removing data points within buffer_radius of the test data.
    # The remaining data is split into training and validation using either a random split or optimization.

    outer_skf = StratifiedKFold(n_splits=5, shuffle=True)
    temp_training_indices, test_indices = [(train, test) for train, test in outer_skf.split(fingerprints, labels)][0]
    test_fingerprints = fingerprints[test_indices]
    test_labels = labels[test_indices]

    # build/remove the buffer

    buffer_radius = 0.4
    temp_training_test_distances = np.min(distances[test_indices, :][:, temp_training_indices], axis=0)
    buffer_indices = np.where(temp_training_test_distances < buffer_radius)[0].tolist()
    temp_training_indices = temp_training_indices[~np.isin(temp_training_indices, buffer_indices)]

    # prep the remaining data for split optimization

    temp_training_fingerprints = fingerprints[temp_training_indices]
    temp_training_labels = labels[temp_training_indices]

    # define how to get performance metrics given a split

    AVE_dataset = ukyDataSet(temp_training_fingerprints, temp_training_labels)
    VE_dataset = ukyDataSet(temp_training_fingerprints, temp_training_labels, AVE=False)

    def model_performance_scores(training_indices, validation_indices):
        training_fingerprints = temp_training_fingerprints[training_indices]
        training_labels = temp_training_labels[training_indices]
        validation_fingerprints = temp_training_fingerprints[validation_indices]
        validation_labels = temp_training_labels[validation_indices]
        validation_weights = AVE_dataset.get_validation_weights(training_indices, validation_indices)

        # random forest metrics
        rf = RandomForestClassifier(n_estimators=100).fit(training_fingerprints, training_labels)
        validation_probs = rf.predict_proba(validation_fingerprints)[:, 1]
        test_probs = rf.predict_proba(test_fingerprints)[:, 1]
        validation_PR_AUC = PR_AUC(validation_probs, validation_labels)
        test_PR_AUC = PR_AUC(test_probs, test_labels)
        weighted_PR_AUC = PR_AUC(validation_probs, validation_labels, validation_weights)

        # nearest neighbor metrics
        nn = KNeighborsClassifier(n_neighbors=1).fit(training_fingerprints, training_labels)
        nn_validation_predictions = nn.predict(validation_fingerprints).astype(bool)
        nn_test_predictions = nn.predict(test_fingerprints).astype(bool)
        validation_nn_F1 = f1_score(validation_labels, nn_validation_predictions)
        test_nn_F1 = f1_score(test_labels, nn_test_predictions)

        # rf-nn similarity
        validation_nn_similarity = nn_similarity(validation_probs, nn_validation_predictions)
        test_nn_similarity = nn_similarity(test_probs, nn_test_predictions)

        # validation AVE bias / VE score
        split = np.isin(np.arange(len(training_indices) + len(validation_indices)), training_indices)
        validation_AVE_bias = AVE_dataset.computeScore(split)[0]
        validation_VE_score = VE_dataset.computeScore(split)[0]

        # test AVE bias / VE score
        training_test_fingerprints = np.vstack([training_fingerprints, fingerprints[test_indices]])
        training_test_split = np.hstack([np.ones(len(training_indices)), np.zeros(len(test_indices))]).astype(bool)
        training_test_labels = np.hstack([training_labels, labels[test_indices]]).astype(bool)
        test_AVE_bias = ukyDataSet(training_test_fingerprints, training_test_labels).computeScore(
            training_test_split, check_valid=False)[0]
        test_VE_score = ukyDataSet(training_test_fingerprints, training_test_labels, AVE=False).computeScore(
            training_test_split, check_valid=False)[0]

        return np.array([len(temp_training_indices), validation_PR_AUC, test_PR_AUC, weighted_PR_AUC,
                         validation_nn_similarity, test_nn_similarity,
                         validation_nn_F1, test_nn_F1,
                         validation_AVE_bias, test_AVE_bias,
                         validation_VE_score, test_VE_score
                         ])

    # split the data using the two optimization methods and record the performance metrics

    AVE_training_indices, AVE_validation_indices = AVE_dataset.geneticOptimizer(1000)
    AVE_perf = model_performance_scores(AVE_training_indices, AVE_validation_indices)

    VE_training_indices, VE_validation_indices = VE_dataset.geneticOptimizer(1000)
    VE_perf = model_performance_scores(VE_training_indices, VE_validation_indices)

    # split the data randomly 5 times and record the mean performance metrics

    inner_skf = StratifiedKFold(n_splits=5, shuffle=True)
    inner_random_splits = [(train, test) for train, test in inner_skf.split(temp_training_fingerprints,
                                                                            temp_training_labels)]
    collect_scores = []
    for random_training_indices, random_validation_indices in inner_random_splits:
        collect_scores.append(model_performance_scores(random_training_indices, random_validation_indices))
    random_perf = np.mean(np.array(collect_scores), axis=0)
    perf_names = ['remainder', 'validation_PR_AUC', 'test_PR_AUC', 'weighted_PR_AUC',
                  'validation_nn_similarity', 'test_nn_similarity',
                  'validation_nn_F1', 'test_nn_F1',
                  'validation_AVE_bias', 'test_AVE_bias',
                  'validation_VE_score', 'test_VE_score']
    column_names = [t + s for t in ['AVE_', 'VE_', 'random_'] for s in perf_names]
    target_results = pd.DataFrame([np.hstack([AVE_perf, VE_perf, random_perf])], columns=column_names)
    target_results.to_csv(output_file, header=True)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        print("Specify decoys file, actives file, and output file.")
