#include "dataloader.h"
#include "linearRegression.h"

int main() {

    // --- Déclaration des pointeurs ---
    double** X       = nullptr;  // matrice complète
    double*  y       = nullptr;  // vecteur cible complet
    int      n_lignes   = 0;
    int      n_colonnes = 0;

    double** X_train = nullptr;  double* y_train = nullptr;
    double** X_test  = nullptr;  double* y_test  = nullptr;
    int      n_train = 0;        int     n_test  = 0;

    double*  moyennes = nullptr;
    double*  ecarts   = nullptr;

    // Étape 1 : chargement du CSV
    chargerCSV("dataset.csv",
               X, y, n_lignes, n_colonnes);

    // Étape 2 : séparation 80% train / 20% test
    separerDonnees(X, y, n_lignes, n_colonnes,
                   X_train, y_train, n_train,
                   X_test,  y_test,  n_test);

    // Étape 3 : normalisation (stats du train uniquement)
    normaliser    (X_train, n_train, n_colonnes, moyennes, ecarts);
    normaliserAvec(X_test,  n_test,  n_colonnes, moyennes, ecarts);

    // Étape 4 : création du modèle via pointeur (Polymorphisme)
    Model* modele = new LinearRegression(0.01, 1000);

    // Étape 5 : entraînement par descente de gradient
    modele->fit(X_train, y_train, n_train, n_colonnes);

    // Étape 6 : évaluation → affiche MSE, RMSE, R²
    modele->score(X_test, y_test, n_test, n_colonnes);
    ::afficher();

    // --- Libération de toute la mémoire ---
    delete modele;
    libererMatrice(X,       n_lignes);
    libererMatrice(X_train, n_train);
    libererMatrice(X_test,  n_test);
    delete[] y;
    delete[] y_train;
    delete[] y_test;
    delete[] moyennes;
    delete[] ecarts;

    return 0;
}