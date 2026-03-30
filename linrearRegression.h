#pragma once
#include <iostream>
#include <cmath>
#include "model.h"
using namespace std;

typedef double** matrice;
typedef double* vecteur;


class linearRegression: public model {
    
    public:
        int n_lignes;
        int n_colonnes;
        double pas;
        int n_iteration;
        int n_lignes_train;
        int n_lignes_test;
        vecteur thetav;

        double* moy;
        double* std_dev;

        linearRegression(int l, int c, double p, int iter) {
            n_lignes     = l;
            n_colonnes   = c;
            pas          = p;
            n_iteration  = iter;
            n_lignes_train = (int)(0.8 * n_lignes);
            n_lignes_test  = n_lignes - n_lignes_train;
            thetav  = new double[n_colonnes + 1];
            moy     = new double[n_colonnes];
            std_dev = new double[n_colonnes];
        }

        virtual vecteur fit(matrice X_training, vecteur Y_training);
        virtual double predict(vecteur X);
        virtual void score(matrice X_test, vecteur Y_test);
        double prod_scal(vecteur v1, vecteur v2, int n);
        vecteur sous_vec(vecteur v1, vecteur v2, int n);
        vecteur multi_vec(double alpha, vecteur v2, int n);
        vecteur multi_matr_vec(matrice mat, vecteur v, int n, int m);
        vecteur multi_matrT_vec(matrice mat, vecteur v, int n, int m);
        vecteur gradient(matrice X, vecteur y);
        vecteur descent_gradient(matrice X, vecteur y);
        void afficher(vecteur v, int n);
        void initialiser_theta();

        void fit_normalisation(matrice X);
        void transform(matrice X, int n);

        ~linearRegression() {
            delete[] thetav;
            delete[] moy;
            delete[] std_dev;
        };
};

double linearRegression::prod_scal(vecteur v1, vecteur v2, int n) {
    double res = 0;
    for (int i = 0; i < n; i++) res += v1[i] * v2[i];
    return res;
}

vecteur linearRegression::sous_vec(vecteur v1, vecteur v2, int n) {
    vecteur res = new double[n];
    for (int i = 0; i < n; i++) res[i] = v1[i] - v2[i];
    return res;
}

vecteur linearRegression::multi_vec(double alpha, vecteur v2, int n) {
    vecteur res = new double[n];
    for (int i = 0; i < n; i++) res[i] = alpha * v2[i];
    return res;
}

vecteur linearRegression::multi_matr_vec(matrice mat, vecteur v, int n, int m) {
    vecteur res = new double[n];
    for (int i = 0; i < n; i++) res[i] = prod_scal(mat[i], v, m);
    return res;
}

vecteur linearRegression::multi_matrT_vec(matrice mat, vecteur v, int n, int m) {
    vecteur res = new double[n];
    for (int i = 0; i < n; i++) {
        res[i] = 0;
        for (int j = 0; j < m; j++) res[i] += mat[j][i] * v[j];
    }
    return res;
}

vecteur linearRegression::gradient(matrice X, vecteur y) {
    vecteur Xtheta = multi_matr_vec(X, thetav, n_lignes_train, n_colonnes + 1);
    vecteur diff   = sous_vec(Xtheta, y, n_lignes_train);
    vecteur res    = multi_matrT_vec(X, diff, n_colonnes + 1, n_lignes_train);
    delete[] Xtheta;
    delete[] diff;
    return res;
}

vecteur linearRegression::descent_gradient(matrice X, vecteur y) {
    for (int i = 0; i < n_iteration; i++) {
        vecteur grad      = gradient(X, y);
        vecteur scaled    = multi_vec(pas / n_lignes_train, grad, n_colonnes + 1);
        vecteur new_theta = sous_vec(thetav, scaled, n_colonnes + 1);
        delete[] grad;
        delete[] scaled;
        delete[] thetav;
        thetav = new_theta;
    }
    return thetav;
}

void linearRegression::afficher(vecteur v, int n) {
    for (int i = 0; i < n; i++) cout << v[i] << endl;
}

void linearRegression::initialiser_theta() {
    for (int i = 0; i <= n_colonnes; i++) thetav[i] = 0;
}

void linearRegression::fit_normalisation(matrice X) {
    for (int j = 0; j < n_colonnes; j++) {
        moy[j]     = 0.0;
        std_dev[j] = 0.0;
        for (int i = 0; i < n_lignes_train; i++) {
            moy[j]     += X[i][j + 1];
            std_dev[j] += X[i][j + 1] * X[i][j + 1];
        }
        moy[j]    /= n_lignes_train;
        std_dev[j] = sqrt(std_dev[j] / n_lignes_train - moy[j] * moy[j]);
        if (std_dev[j] == 0) std_dev[j] = 1;
    }
    for (int i = 0; i < n_lignes_train; i++)
        for (int j = 0; j < n_colonnes; j++)
            X[i][j + 1] = (X[i][j + 1] - moy[j]) / std_dev[j];
}

void linearRegression::transform(matrice X, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n_colonnes; j++)
            X[i][j + 1] = (X[i][j + 1] - moy[j]) / std_dev[j];
}

vecteur linearRegression::fit(matrice X_training, vecteur Y_training) {
    return descent_gradient(X_training, Y_training);
}

double linearRegression::predict(vecteur X) {
    return prod_scal(X, thetav, n_colonnes + 1);
}

void linearRegression::score(matrice X_test, vecteur Y_test) {
    double r = 0, m = 0, r1 = 0;
    for (int j = 0; j < n_lignes_test; j++) m += Y_test[j] / n_lignes_test;
    for (int i = 0; i < n_lignes_test; i++) {
        r  += pow(predict(X_test[i]) - Y_test[i], 2);
        r1 += pow(m - Y_test[i], 2);
    }
    cout << "La valeur de R-carre est : " << 1 - r / r1 << endl;
}