#pragma once
#include <fstream>
#include <sstream>
#include <iostream>
using namespace std;

typedef double** matrice;
typedef double* vecteur;

void lire_csv(matrice& data, string file_name, int n_lignes, int n_colonnes) {
    data = new double*[n_lignes];
    for (int i = 0; i < n_lignes; i++) {
        data[i] = new double[n_colonnes + 1];
        data[i][0] = 1;
    }

    ifstream file(file_name);

    if (!file.is_open()) {
        cerr << "Erreur : impossible d'ouvrir le fichier " << file_name << endl;
        return;
    }

    string line;
    int i = 0;
    while (std::getline(file, line)) {
        stringstream ss(line);
        string value;
        int j = 1;
        while (getline(ss, value, ',')) {
            data[i][j] = stod(value);
            j++;
        }
        i++;
    }
    file.close();
}

void split_data(matrice data, int n_lignes, int n_colonnes,
                matrice& X_training, matrice& X_test,
                vecteur& Y_training, vecteur& Y_test) {

    int n_lignes_train = (int)(0.8 * n_lignes);
    int n_lignes_test  = n_lignes - n_lignes_train;

    X_training = new double*[n_lignes_train];
    for (int i = 0; i < n_lignes_train; i++) {
        X_training[i] = new double[n_colonnes + 1];
    }

    X_test = new double*[n_lignes_test];
    for (int i = 0; i < n_lignes_test; i++) {
        X_test[i] = new double[n_colonnes + 1];
    }

    Y_training = new double[n_lignes_train];
    Y_test     = new double[n_lignes_test];

    for (int i = 0; i < n_lignes_train; i++) {
        X_training[i][0] = 1;
        for (int j = 1; j <= n_colonnes; j++) {
            X_training[i][j] = data[i][j];
        }
        Y_training[i] = data[i][n_colonnes];
    }

    for (int i = 0; i < n_lignes_test; i++) {
        X_test[i][0] = 1;
        for (int j = 1; j <= n_colonnes; j++) {
            X_test[i][j] = data[n_lignes_train + i][j];
        }
        Y_test[i] = data[n_lignes_train + i][n_colonnes];
    }
}