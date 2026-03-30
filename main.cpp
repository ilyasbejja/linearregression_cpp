#include "model.h"
#include "dataset.h"
#include "linrearRegression.h"
#include <iostream>
using namespace std;

int main() {
    int n_lignes    = 10000;
    int n_colonnes  = 6;
    int n_iteration = 1000;
    double pas      = 0.1;

    matrice data;
    matrice X_training, X_test;
    vecteur Y_training, Y_test;

    lire_csv(data, "dataset.csv", n_lignes, n_colonnes);
    split_data(data, n_lignes, n_colonnes, X_training, X_test, Y_training, Y_test);

    linearRegression lr(n_lignes, n_colonnes, pas, n_iteration);

    lr.fit_normalisation(X_training);
    lr.transform(X_test, lr.n_lignes_test);

    lr.initialiser_theta();
    lr.fit(X_training, Y_training);

    lr.score(X_test, Y_test);

    for (int i = 0; i < n_lignes; i++)        delete[] data[i];
    delete[] data;
    for (int i = 0; i < lr.n_lignes_train; i++) delete[] X_training[i];
    delete[] X_training;
    for (int i = 0; i < lr.n_lignes_test; i++)  delete[] X_test[i];
    delete[] X_test;
    delete[] Y_training;
    delete[] Y_test;

    return 0;
}