#include "linearRegression.h"
#include <iostream>
#include <cmath>
using namespace std;

// -----------------------------------------------
// Utilitaires mathématiques internes
// -----------------------------------------------

// Produit scalaire : Σ a[i] * b[i]
double LinearRegression::produitScalaire(double* a,
                                          double* b,
                                          int n) const {
    double resultat = 0.0;
    for (int i = 0; i < n; i++)
        resultat += a[i] * b[i];
    return resultat;
}

// Moyenne d'un tableau
double LinearRegression::moyenne(double* v, int n) const {
    double somme = 0.0;
    for (int i = 0; i < n; i++)
        somme += v[i];
    return somme / n;
}

// Produit X (n×p) × v (p) → tableau de taille n
double* LinearRegression::produitMatVec(double** X, double* v,
                                         int n_lignes,
                                         int n_colonnes) const {
    double* resultat = new double[n_lignes]();
    for (int i = 0; i < n_lignes; i++)
        resultat[i] = produitScalaire(X[i], v, n_colonnes);
    return resultat;
}

// Produit X^T (p×n) × v (n) → tableau de taille p
double* LinearRegression::produitXtVec(double** X, double* v,
                                        int n_lignes,
                                        int n_colonnes) const {
    double* resultat = new double[n_colonnes]();
    for (int j = 0; j < n_colonnes; j++)
        for (int i = 0; i < n_lignes; i++)
            resultat[j] += X[i][j] * v[i];
    return resultat;
}

// -----------------------------------------------
// fit — Entraînement par descente de gradient
// -----------------------------------------------
void LinearRegression::fit(double** X, double* y,
                            int n_lignes, int n_colonnes) {

    // Initialisation des coefficients à 0
    delete[] coefficients;
    coefficients = new double[n_colonnes]();  // alloue et initialise à 0
    intercept    = 0.0;

    for (int iter = 0; iter < nb_iterations; iter++) {

        // Étape 1 : prédictions → ŷ = X * θ + intercept
        double* y_hat = produitMatVec(X, coefficients,
                                      n_lignes, n_colonnes);
        for (int i = 0; i < n_lignes; i++)
            y_hat[i] += intercept;

        // Étape 2 : résidus → erreur = ŷ - y
        double* erreur = new double[n_lignes];
        for (int i = 0; i < n_lignes; i++)
            erreur[i] = y_hat[i] - y[i];

        // Étape 3 : gradient des coefficients = (1/n) * X^T * erreur
        double* grad_coef = produitXtVec(X, erreur,
                                          n_lignes, n_colonnes);
        for (int j = 0; j < n_colonnes; j++)
            grad_coef[j] /= n_lignes;

        // Étape 3 : gradient de l'intercept = (1/n) * Σ erreur
        double grad_intercept = moyenne(erreur, n_lignes);

        // Étape 4 : mise à jour des paramètres
        for (int j = 0; j < n_colonnes; j++)
            coefficients[j] -= taux_apprentissage * grad_coef[j];
        intercept -= taux_apprentissage * grad_intercept;

        // Affichage MSE toutes les 100 itérations
        

        // Libération des tableaux temporaires
        delete[] y_hat;
        delete[] erreur;
        delete[] grad_coef;
    }
}

// -----------------------------------------------
// predict — Prédiction pour un exemple x
// -----------------------------------------------
double LinearRegression::predict(double* x, int n_colonnes) const {
    // ŷ = intercept + θ^T * x
    return intercept + produitScalaire(coefficients, x, n_colonnes);
}

// -----------------------------------------------
// score — Affiche MSE, RMSE, R² dans le terminal
// -----------------------------------------------
void LinearRegression::score(double** X, double* y,
                              int n_lignes, int n_colonnes) const {

    // Calcul de toutes les prédictions
    double* predictions = produitMatVec(X, coefficients,
                                        n_lignes, n_colonnes);
    for (int i = 0; i < n_lignes; i++)
        predictions[i] += intercept;

    // SS_res = Σ (y - ŷ)²
    double ss_res = 0.0;
    for (int i = 0; i < n_lignes; i++)
        ss_res += pow(y[i] - predictions[i], 2);

    // SS_tot = Σ (y - ȳ)²
    double y_moy = moyenne(y, n_lignes);
    double ss_tot = 0.0;
    for (int i = 0; i < n_lignes; i++)
        ss_tot += pow(y[i] - y_moy, 2);

    // Métriques
    double mse  = ss_res / n_lignes;
    double rmse = sqrt(mse);
    double r2   = 1.0 - (ss_res / ss_tot);

    cout << "\n===== Evaluation du Modele =====" << endl;
    cout << "MSE  = " << mse  << endl;
    cout << "RMSE = " << rmse << endl;
    cout << "R2   = " << r2   << endl;
    cout << "================================\n" << endl;
    

    delete[] predictions;
}


void LinearRegression::afficher() {

    cout << "Theta " << 0 << intercept << endl;
    for ( int i=0; i<4; i++) {
        cout << "Theta " << i+1 << coefficients[i] << endl;
    }
}