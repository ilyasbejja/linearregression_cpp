#include "dataloader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cmath>
using namespace std;

// -----------------------------------------------
// Fonction utilitaire : compte les lignes du CSV (hors en-tête)
// -----------------------------------------------
static int compterLignes(const string& chemin) {
    ifstream f(chemin);
    int count = 0;
    string ligne;
    while (getline(f, ligne)) count++;
    return count - 1;  // -1 pour l'en-tête
}

// -----------------------------------------------
// Fonction utilitaire : compte les colonnes du CSV
// -----------------------------------------------
static int compterColonnes(const string& chemin) {
    ifstream f(chemin);
    string ligne;
    getline(f, ligne);  // ignorer l'en-tête
    getline(f, ligne);  // première ligne de données
    int count = 0;
    stringstream ss(ligne);
    string cellule;
    while (getline(ss, cellule, ',')) count++;
    return count;
}

// -----------------------------------------------
// chargerCSV — remplit X (double**) et y (double*)
// -----------------------------------------------
void chargerCSV(const string& chemin,
                double**& X, double*& y,
                int& n_lignes, int& n_colonnes) {

    n_lignes  = compterLignes(chemin);
    int total = compterColonnes(chemin);
    n_colonnes = total - 1;  // dernière colonne = y

    // Allocation dynamique de X et y
    X = new double*[n_lignes];
    for (int i = 0; i < n_lignes; i++)
        X[i] = new double[n_colonnes]();
    y = new double[n_lignes]();

    ifstream fichier(chemin);
    if (!fichier.is_open())
        throw runtime_error("Impossible d'ouvrir : " + chemin);

    string ligne;
    bool en_tete = true;
    int i = 0;

    while (getline(fichier, ligne)) {

        // Ignorer la première ligne (en-tête)
        if (en_tete) { en_tete = false; continue; }

        stringstream ss(ligne);
        string cellule;

        // Lire toutes les colonnes de la ligne
        double* ligne_tmp = new double[total];
        for (int j = 0; j < total; j++) {
            getline(ss, cellule, ',');
            ligne_tmp[j] = stod(cellule);  // string → double
        }

        // Dernière colonne → y, le reste → X[i]
        y[i] = ligne_tmp[total - 1];
        for (int j = 0; j < n_colonnes; j++)
            X[i][j] = ligne_tmp[j];

        delete[] ligne_tmp;
        i++;
    }
}

// -----------------------------------------------
// separerDonnees — split train / test
// -----------------------------------------------
void separerDonnees(double** X,       double*  y,
                    int n_lignes,     int n_colonnes,
                    double**& X_train, double*& y_train, int& n_train,
                    double**& X_test,  double*& y_test,  int& n_test,
                    double ratio_train) {

    n_train = static_cast<int>(n_lignes * ratio_train);  // ex: 8000
    n_test  = n_lignes - n_train;                         // ex: 2000

    // Allocation des ensembles train et test
    X_train = new double*[n_train];
    for (int i = 0; i < n_train; i++) {
        X_train[i] = new double[n_colonnes];
        for (int j = 0; j < n_colonnes; j++)
            X_train[i][j] = X[i][j];  // copie ligne par ligne
    }
    y_train = new double[n_train];
    for (int i = 0; i < n_train; i++)
        y_train[i] = y[i];

    X_test = new double*[n_test];
    for (int i = 0; i < n_test; i++) {
        X_test[i] = new double[n_colonnes];
        for (int j = 0; j < n_colonnes; j++)
            X_test[i][j] = X[n_train + i][j];  // copie à partir de n_train
    }
    y_test = new double[n_test];
    for (int i = 0; i < n_test; i++)
        y_test[i] = y[n_train + i];
}

// -----------------------------------------------
// normaliser — normalise X_train et sauvegarde les stats
// -----------------------------------------------
void normaliser(double** X, int n_lignes, int n_colonnes,
                double*& moyennes, double*& ecarts) {

    moyennes = new double[n_colonnes]();
    ecarts   = new double[n_colonnes]();

    // Calcul de la moyenne par colonne
    for (int j = 0; j < n_colonnes; j++) {
        for (int i = 0; i < n_lignes; i++)
            moyennes[j] += X[i][j];
        moyennes[j] /= n_lignes;
    }

    // Calcul de l'écart-type par colonne
    for (int j = 0; j < n_colonnes; j++) {
        for (int i = 0; i < n_lignes; i++)
            ecarts[j] += pow(X[i][j] - moyennes[j], 2);
        ecarts[j] = sqrt(ecarts[j] / n_lignes);
        if (ecarts[j] == 0) ecarts[j] = 1.0;  // éviter division par zéro
    }

    // Application : (x - moyenne) / écart-type
    for (int i = 0; i < n_lignes; i++)
        for (int j = 0; j < n_colonnes; j++)
            X[i][j] = (X[i][j] - moyennes[j]) / ecarts[j];
}

// -----------------------------------------------
// normaliserAvec — applique les stats du train sur le test
// -----------------------------------------------
void normaliserAvec(double** X, int n_lignes, int n_colonnes,
                    double* moyennes, double* ecarts) {
    for (int i = 0; i < n_lignes; i++)
        for (int j = 0; j < n_colonnes; j++)
            X[i][j] = (X[i][j] - moyennes[j]) / ecarts[j];
}

// -----------------------------------------------
// libererMatrice — libère la mémoire d'une matrice double**
// -----------------------------------------------
void libererMatrice(double** X, int n_lignes) {
    for (int i = 0; i < n_lignes; i++)
        delete[] X[i];
    delete[] X;
}