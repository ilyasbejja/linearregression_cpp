#pragma once
#include <iostream>
using namespace std;
typedef double** matrice;
typedef double* vecteur;

class model {
    protected:
        int n_lignes;
        int n_colonnes;

    public:
        virtual vecteur fit(matrice X_training, vecteur Y_training) { return nullptr; }
        virtual double predict(vecteur X) { return 0.0; }
        virtual void score(matrice X_test, vecteur Y_test) {}
};