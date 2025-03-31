// Object-Oriented Implementation of 1D and 2D Ising Simulators
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <random>
#include <iomanip>

using namespace std;

//  Base IsingModel Class (common configuration) 
class IsingModel {
protected:
    double beta;
    int steps;
    int burn_in;
    int sample_gap;
    mt19937 gen;
    uniform_real_distribution<> rand01;

public:
    IsingModel(double b, int s, int burn, int gap)
        : beta(b), steps(s), burn_in(burn), sample_gap(gap), rand01(0.0, 1.0) {
        random_device rd;
        gen.seed(rd());
    }
    virtual void simulate(ofstream& file) = 0;  // Pure virtual function (implemented by subclasses)
};

// One-Dimensional Ising Model 
class Ising1D : public IsingModel {
private:
    int N;
    vector<int> spins;

    double compute_energy() {
        double E = 0.0;
        for (int i = 0; i < N; ++i)
            E -= spins[i] * spins[(i + 1) % N];
        return E;
    }

    double compute_magnetization() {
        double M = 0.0;
        for (int s : spins) M += s;
        return M;
    }

public:
    Ising1D(int n, double b, int s, int burn, int gap)
        : IsingModel(b, s, burn, gap), N(n), spins(n, 1) {}

    void simulate(ofstream& file) override {
        uniform_int_distribution<> dist(0, N - 1);
        for (int step = 0; step < steps; ++step) {
            for (int i = 0; i < N; ++i) {
                int idx = dist(gen);
                int left = spins[(idx - 1 + N) % N];
                int right = spins[(idx + 1) % N];
                int dE = 2 * spins[idx] * (left + right);
                if (dE <= 0 || rand01(gen) < exp(-beta * dE))
                    spins[idx] *= -1;
            }
            if (step >= burn_in && (step - burn_in) % sample_gap == 0) {
                file << beta << "," << step << "," << compute_energy()
                     << "," << compute_magnetization() << "\n";
            }
        }
    }
};

// Two-Dimensional Ising Model 
class Ising2D : public IsingModel {
private:
    int N;
    vector<vector<int>> spins;

    double compute_energy() {
        double E = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j) {
                int right = spins[i][(j + 1) % N];
                int down = spins[(i + 1) % N][j];
                E -= spins[i][j] * (right + down);
            }
        return E;
    }

    double compute_magnetization() {
        double M = 0.0;
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                M += spins[i][j];
        return M;
    }

public:
    Ising2D(int n, double b, int s, int burn, int gap)
        : IsingModel(b, s, burn, gap), N(n), spins(n, vector<int>(n, 1)) {}

    void simulate(ofstream& file) override {
        uniform_int_distribution<> dist(0, N - 1);
        for (int step = 0; step < steps; ++step) {
            for (int k = 0; k < N * N; ++k) {
                int i = dist(gen);
                int j = dist(gen);
                int sum_neigh = spins[(i + 1) % N][j] + spins[(i - 1 + N) % N][j] +
                                spins[i][(j + 1) % N] + spins[i][(j - 1 + N) % N];
                int dE = 2 * spins[i][j] * sum_neigh;
                if (dE <= 0 || rand01(gen) < exp(-beta * dE))
                    spins[i][j] *= -1;
            }
            if (step >= burn_in && (step - burn_in) % sample_gap == 0) {
                file << beta << "," << step << "," << compute_energy()
                     << "," << compute_magnetization() << "\n";
            }
        }
    }
};

// Main Function 
int main() {
    vector<double> betas = {0.1, 0.2, 0.3, 0.4, 0.5,
                            0.6, 0.7, 0.8, 0.9, 1.0};
    int steps = 10000; // set steps
    int burn_in = 1000;// set burn_in
    int sample_gap = 10;

    ofstream out1d("ising_1d.csv");
    ofstream out2d("ising_2d.csv");
    out1d << "Beta,Step,Energy,Magnetization\n";
    out2d << "Beta,Step,Energy,Magnetization\n";

    for (double beta : betas) {
        cout << "Running beta = " << beta << "..." << endl;

        Ising1D model1d(100, beta, steps, burn_in, sample_gap);
        model1d.simulate(out1d);//100

        Ising2D model2d(100, beta, steps, burn_in, sample_gap);
        model2d.simulate(out2d);//100*100
    }

    out1d.close();
    out2d.close();
    cout << " Simulation complete. Data saved.\n";
    return 0;
}