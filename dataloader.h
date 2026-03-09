#ifndef DATALOADER_H
#define DATALOADER_H

#include <string>

// -----------------------------------------------
// Charge un fichier CSV :
// - alloue dynamiquement X (double**) et y (double*)
// - remplit n_lignes et n_colonnes
// - première ligne ignorée (en-tête)
// - dernière colonne → y, le reste → X
// -----------------------------------------------
void chargerCSV(const std::string& chemin,
                double**& X,   // matrice allouée ici
                double*&  y,   // vecteur alloué ici
                int& n_lignes,
                int& n_colonnes);

// -----------------------------------------------
// Sépare X et y en ensembles train et test
// ratio_train = 0.8 → 80% train, 20% test
// -----------------------------------------------
void separerDonnees(double** X,       double*  y,
                    int n_lignes,     int n_colonnes,
                    double** &X_train, double* &y_train, int& n_train,
                    double** &X_test,  double* &y_test,  int& n_test,
                    double ratio_train = 0.8);

// -----------------------------------------------
// Normalise X_train : (x - moyenne) / ecart-type
// Sauvegarde moyennes et ecarts pour le test
// -----------------------------------------------
void normaliser(double** X,     int n_lignes, int n_colonnes,
                double*& moyennes,
                double*& ecarts);

// Applique la normalisation du train sur le test
void normaliserAvec(double** X,    int n_lignes, int n_colonnes,
                    double*  moyennes,
                    double*  ecarts);

// -----------------------------------------------
// Libère la mémoire allouée pour une matrice
// -----------------------------------------------
void libererMatrice(double** X, int n_lignes);

#endif