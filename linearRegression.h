#ifndef LINEARREGRESSION_H
#define LINEARREGRESSION_H

#include "model.h"

// -----------------------------------------------
// Régression linéaire par descente de gradient
// Héritage de Model (POO)
// -----------------------------------------------
class LinearRegression : public Model {

private:
    double  intercept;          // biais θ₀
    double* coefficients;       // tableau θ₁...θp (alloué dynamiquement)
    double  taux_apprentissage; // learning rate
    int     nb_iterations;      // nombre d'itérations

    // --- Utilitaires mathématiques internes ---

    // Produit scalaire : Σ a[i] * b[i]
    double produitScalaire(double* a, double* b, int n) const;

    // Calcule la moyenne d'un tableau
    double moyenne(double* v, int n) const;

    // Multiplie X (n×p) par v (p) → résultat (n) alloué dynamiquement
    double* produitMatVec(double** X, double* v,
                          int n_lignes, int n_colonnes) const;

    // Calcule X^T (p×n) × v (n) → résultat (p) alloué dynamiquement
    double* produitXtVec(double** X, double* v,
                         int n_lignes, int n_colonnes) const;

public:
    // Constructeur : initialise les hyperparamètres
    LinearRegression(double pas = 0.01, int iters = 1000)
        : taux_apprentissage(pas), nb_iterations(iters),
          intercept(0.0), coefficients(nullptr) {}

    // Destructeur : libère les coefficients
    ~LinearRegression() { delete[] coefficients; }

    // Entraînement par descente de gradient
    void fit(double** X, double* y,
             int n_lignes, int n_colonnes)          override;

    // Prédiction pour un exemple x
    double predict(double* x, int n_colonnes) const override;

    // Évaluation : affiche MSE, RMSE, R²
    void score(double** X, double* y,
               int n_lignes, int n_colonnes)  const override;
    
    void afficher();
    
};

#endif