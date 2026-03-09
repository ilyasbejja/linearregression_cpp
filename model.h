#ifndef MODEL_H
#define MODEL_H

// -----------------------------------------------
// Classe abstraite — contrat commun à tout modèle
// Un vecteur = double*  avec sa taille
// Une matrice = double** avec ses lignes et colonnes
// -----------------------------------------------
class Model {
public:
    // Entraînement du modèle
    virtual void fit(double** X, double* y,
                     int n_lignes, int n_colonnes)          = 0;

    // Prédiction pour un seul exemple x
    virtual double predict(double* x, int n_colonnes) const = 0;

    // Évaluation : affiche MSE, RMSE, R² dans le terminal
    virtual void score(double** X, double* y,
                       int n_lignes, int n_colonnes)  const = 0;
    
    // Destructeur virtuel
    virtual ~Model() {}
    
};

#endif