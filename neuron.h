#ifndef NEURON_H
#define NEURON_H
#include <vector>
#include <time.h>
#include <cstdlib>
#include <utility>
#include <math.h>

using namespace std;

class Neuron {
private:
	//parameters
	double value;
	double h;
	vector <double> weights;
	vector <double> lastWeights;

public:
	Neuron() { value = 0; h = 0; }
	Neuron(int count);
	Neuron(vector <double> weights);
	void calculateH(double l);
	void addValue(double v) { value += v; }
	void updateWeight(double weight, int neuron) {
		lastWeights[neuron] = weights[neuron];
		weights[neuron] += weight;
	}

	/*
	Calculates the difference in weights
	between the last run and this one
	*/
	double getDiff(int neuron) { return weights[neuron] - lastWeights[neuron]; }

	//sets
	void setValue(double v) { value = v; }
	void setH(double y) { h = y; }
	//gets
	double getValue() { return value; }
	double getH() { return h; }
	double getWeight(int neuron) { return weights[neuron]; }
};

/*
Constructor. Creates a neuron with
the specified ammount of weights
*/
Neuron::Neuron(int count) {
	//create a neuron with 'count' 
	//random weights
	value = 0;
	h = 1;	//setting a value of 1 here ensures that bias is always handled as 1
	double w;
	for (int i = 0; i<count; i++) {
		w = static_cast <double>(rand()) / (static_cast <double> (RAND_MAX));
		weights.push_back(w);
		lastWeights.push_back(w);
	}
}

Neuron::Neuron(vector <double> vect) {
	value = 0;
	h = 1;
	weights = vect;
	lastWeights = vect;
}

/*
This method calculates firing
strength for the neuron using
the sigmoid function
*/
void Neuron::calculateH(double l) {
	h = 1 / (1 + pow(2.718, -l*value));
	value = 0;
}

#endif