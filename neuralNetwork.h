#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <map>
#include "neuron.h"

using namespace std;

class NeuralNetwork {
private:
	//network parameters
	double lambda;
	double etha;
	double mommentum;
	vector <pair <pair <double, double>, pair <double, double> > > originalData	;
	vector <vector <double > > normalizer;

	//network neurons
	vector <Neuron> inputNeurons;
	vector <vector <Neuron> > hiddenLayers;
	vector <Neuron> outputNeurons;

public:
	NeuralNetwork(double l, double e, double m,
		int inputCount, int layersCount,
		int hiddenCount, int outputCount);
	void run(int epochs);
	void runInputs(const vector <double> &input);
	void runHiddenLayer();
	void runOutputLayer();
	vector <vector <double> > onlineTraining(double a, double b);
	vector <double> calculateError(const vector <double> &output);
	vector <double> calculateOutputGradients(const vector <double> &error);
	vector <double> calculateInputGradients(const vector <double> &outputGradients);
	void updateOutputWeights(const vector <double> &outputGradients);
	void updateInputWeights(const vector <double> &inputGradients, const vector <double> &input);
	double runValidation();
	vector <vector <double> >normalizeValues(string file);

};

/*
Constructor. Sets the values and creates the
specified ammount of neurons
*/
NeuralNetwork::NeuralNetwork(double l, double e, double m,
	int inputCount, int layersCount,
	int hiddenCount, int outputCount) {
	lambda = l; etha = e; mommentum = m;

	//create input neurons and add them to the vector
	//add an extra neuron to act as bias
	for (int i = 0; i < inputCount + 1; i++) {
		Neuron n(hiddenCount);
		inputNeurons.push_back(n);
	}

	//creat hidden layers and neurons and 
	//add them to the vector, add an extra 
	//neuron to act as bias
	for (int i = 0; i<layersCount; i++) {
		vector <Neuron> layer;
		for (int j = 0; j<hiddenCount + 1; j++) {
			Neuron n(outputCount);
			layer.push_back(n);
		}
		hiddenLayers.push_back(layer);
	}

	//create output neurons and add them to the vector
	for (int i = 0; i<outputCount; i++) {
		Neuron n;
		outputNeurons.push_back(n);
	}
}


/*
This method runs the network over the training
set for the specified ammount of epochs
*/
void NeuralNetwork::run(int epochs) {
	ifstream file;
	ofstream outFile;
	ofstream outValidationFile;
	double sumError;
	
	//creates the normalizing vector
	normalizer = normalizeValues("dataNN.txt");
	outFile.open("errorEpochTraining.txt");
	outValidationFile.open("errorEpochValidation.txt");
	for (int i = 0; i<epochs; i++) {
		double avgError = 0;
		double count = 0;
		double countBad = 0;
		double otherValue;
		sumError = 0;
		file.open("dataNN.txt");
		if (!file.is_open()) {
			cout << "File could not be open\n";
			exit(0);
		}
		while (!file.eof()) {
			vector <double> input;
			vector <double> output;
			//bias input
			input.push_back(1);
			double value;
			double dValue;
			//read input and output and normalize it			
			for (int j = 1; j<inputNeurons.size(); j++) {
				file >> value;
				value = (value - normalizer[j - 1][0]) / (normalizer[j - 1][1] - normalizer[j - 1][0]);
				input.push_back(value);
				//input.push_back(value);
			}
			for (int j = 0; j<outputNeurons.size(); j++) {
				file >> dValue;
				dValue = (dValue - normalizer[j+2][0]) / (normalizer[j+2][1] - normalizer[j+2][0]);
				output.push_back(dValue);
			}

			//go through the input neurons	
			runInputs(input);

			//go through the output neurons
			runOutputLayer();

			//calculate errors
			vector <double> error = calculateError(output);

			//calculate gradients
			vector <double> outputGradients = calculateOutputGradients(error);

			//calculate input gradients
			vector <double> inputGradients = calculateInputGradients(outputGradients);

			//calculate weight updating for the
			//weights going towards output neurons
			updateOutputWeights(outputGradients);

			//calculate weight updating for the
			//weights going towards output neurons
			updateInputWeights(inputGradients, input);
			
			//calculate the average error
			avgError += (error[0])*(error[0]);
			avgError += (error[1])*(error[1]);
			count++;
		}

		outFile << "Epoch "<<i<<" error: "<< avgError / count  << endl;
		
		//error per epoch
		file.close();
		//validation
		double validationError = runValidation();
		outValidationFile << "Epoch " << i << " error: " << validationError << endl;
	}
	outFile.close();
	outValidationFile.close();
}

/*
This method runs the input neurons.
It receives a vector containing the
input already normalized.
*/
void NeuralNetwork::runInputs(const vector <double> &input) {
	//For every neuron in the first layer of
	//the hidden layer go through all the
	//input neurons connecting to it.
	for (int j = 1; j<(hiddenLayers[0]).size(); j++) {
		for (int k = 0; k<inputNeurons.size(); k++) {
			hiddenLayers[0][j].addValue(inputNeurons[k].getWeight(j-1)*input[k]);
		}
		hiddenLayers[0][j].calculateH(lambda);
	}
}


/*
This method would be used in case there
are more than one hidden layers in the
network.
*/
void NeuralNetwork::runHiddenLayer() {
	//loop through every layer except the 
	//first one as it has already been done
	for (int j = 1; j<hiddenLayers.size(); j++) {
		for (int k = 1; k<hiddenLayers[j].size(); k++) {
			for (int m = 0; m<hiddenLayers[j - 1].size(); m++) {
				hiddenLayers[j][k].addValue(hiddenLayers[j - 1][m].getWeight(k)*hiddenLayers[j - 1][m].getH());
			}
			hiddenLayers[j][k].calculateH(lambda);
		}
	}
}

/*
This method runs the output 
layer of the neural network.
*/
void NeuralNetwork::runOutputLayer() {
	int m = hiddenLayers[hiddenLayers.size() - 1].size();
	//For every output neuron go through
	//every hidden neuron connecting to it
	for (int j = 0; j<outputNeurons.size(); j++) {
		for (int k = 0; k<m; k++) {
			outputNeurons[j].addValue(hiddenLayers[0][k].getWeight(j)*hiddenLayers[0][k].getH());
		}
		outputNeurons[j].calculateH(lambda);
	}
}

/*
This method runs the online trainig of the neural network.
It receives two input values which are run through the neuron.
The error is calculated by comparing the output to the closest
pair of inputs found in the training set.
*/
vector <vector <double> >NeuralNetwork::onlineTraining(double a, double b) {
	vector <double> inp;
	vector <vector <double> > outputVect;
	vector <double> speeds;
	vector <double> error(2);
	//bias input
	inp.push_back(1);
	//make sure the values do not exceed the maximum values
	//found in the training data
	if (a > normalizer[0][1])
		a = normalizer[0][1];
	if (b > normalizer[1][1])
		b = normalizer[1][1];
	//normalize values
	inp.push_back((a - normalizer[0][0]) / (normalizer[0][1] - normalizer[0][0]));
	inp.push_back((b - normalizer[1][0]) / (normalizer[1][1] - normalizer[1][0]));

	//run the input nodes
	runInputs(inp);

	//run the output nodes
	runOutputLayer();
	
	//de-normalize output values
	double denormalizedL, denormalizedR;
	denormalizedL = outputNeurons[0].getH()*(normalizer[2][1] - normalizer[2][0]) + normalizer[2][0];
	denormalizedR = outputNeurons[1].getH()*(normalizer[3][1] - normalizer[3][0]) + normalizer[3][0];

	//look for closest value
	int position, i;
	double minDistance=10000000000;
	double distance;
	for (i = 0; i< originalData.size(); i++) {
		distance = pow(pow(originalData[i].first.first - a, 2.0) + pow(originalData[i].first.second - b, 2.0), 0.5);
		if (distance < minDistance) {
			minDistance = distance;
			position = i;
		}
	}

	//look for multiple entries
	i = position;
	double minimum = 100000000000;
	double speedDistance;
	do {
		distance = pow(pow(originalData[i].first.first - a, 2.0) + pow(originalData[i].first.second - b, 2.0), 0.5);
		speedDistance = pow(pow(originalData[i].second.first - denormalizedL, 2.0) + pow(originalData[i].second.second - denormalizedR, 2.0), 0.5);
		if (speedDistance < minimum) {
			minimum = speedDistance;
			position = i;
		}
		i++;
	} while (distance == minDistance && i<originalData.size());
	double normalizedOutputLeft = (originalData[position].second.first - normalizer[2][0]) / (normalizer[2][1] - normalizer[2][0]);
	double normalizedOutputRight = (originalData[position].second.second - normalizer[3][0]) / (normalizer[3][1] - normalizer[3][0]);
	error[0] = pow(normalizedOutputLeft - outputNeurons[0].getH(), 2.0);
	error[1] = pow(normalizedOutputRight - outputNeurons[1].getH(), 2.0);

	//push back values to the return vector
	speeds.push_back(denormalizedL);
	speeds.push_back(denormalizedR);
	outputVect.push_back(speeds);
	outputVect.push_back(error);
	return outputVect;
}

/*
This method calculates the error of the run.
It receives a vector containing the output
and it compares it to the expected one.
*/
vector <double> NeuralNetwork::calculateError(const vector <double> &output) {
	vector <double> error;
	for (int j = 0; j < outputNeurons.size(); j++) {
		error.push_back(output[j] - outputNeurons[j].getH());
	}
	return error;
}

/*
This method calculates the gradient of the
output neurons. It receives a vector containing
the error at each neuron.
*/
vector <double> NeuralNetwork::calculateOutputGradients(const vector <double> &error) {
	double gradient;
	vector <double> outputGradients;
	//For every node in the output layer
	//calculate the gradient
	for (int j = 0; j < outputNeurons.size(); j++) {
		gradient = lambda*outputNeurons[j].getH()*(1 - outputNeurons[j].getH())*error[j];
		outputGradients.push_back(gradient);
	}
	return outputGradients;
}

/*
This method calculates the gradients for the input nodes.
It receives a vector containing the gradients for the last
layer of the hidden layers.
*/
vector <double> NeuralNetwork::calculateInputGradients(const vector <double> &outputGradients) {
	double sum = 0;
	double gradient;
	vector <double> inputGradients;
	//For every node in the hidden layer
	//calculate the gradient 
	for (int j = 1; j < hiddenLayers[0].size(); j++) {
		//Sum of Gk*Wki
		for (int k = 0; k < outputGradients.size(); k++) {
			sum += outputGradients[k] * hiddenLayers[0][j].getWeight(k);
		}
		gradient = lambda*(hiddenLayers[0][j].getH())*(1 - hiddenLayers[0][j].getH())*sum;
		sum = 0;
		inputGradients.push_back(gradient);
	}
	return inputGradients;
}

/*
This method updates the weights of the last hidden layer
neurons. It receives a vector containing the corresponding
gradients
*/
void NeuralNetwork::updateOutputWeights(const vector <double> &outputGradients) {
	double deltaWeight;
	double alphaChange;
	//For every node in the output layer
	//update the weight of all the hidden 
	//nodes going towards it
	for (int j = 0; j < outputNeurons.size(); j++) {
		for (int k = 0; k <hiddenLayers[0].size(); k++) {
			alphaChange = mommentum*hiddenLayers[0][k].getDiff(j);
			deltaWeight = etha*outputGradients[j] * hiddenLayers[0][k].getH() + alphaChange;
			hiddenLayers[0][k].updateWeight(deltaWeight, j);
		}
	}
}

/*
This method updates the weights of the input neurons.
It receives a vector containing the corresponding
gradients and a vector with the original inputs
*/
void NeuralNetwork::updateInputWeights(const vector <double> &inputGradients, const vector <double> &input) {
	double deltaWeight;
	double alphaChange;
	//For every node in the hidden layer
	//udate the weight of all the input
	//nodes going towards it
	for (int j = 1; j < hiddenLayers[0].size(); j++) {
		for (int k = 0; k < inputNeurons.size(); k++) {
			alphaChange = mommentum*inputNeurons[k].getDiff(j - 1);
			deltaWeight = etha*inputGradients[j - 1] * input[k] + alphaChange;
			inputNeurons[k].updateWeight(deltaWeight, j - 1);
		}
	}
}

/*
This method runs the network over the validation data
*/
double NeuralNetwork::runValidation() {
	ifstream file;
	ofstream outFile;
	double sumError = 0;
	double count = 0;
	file.open("validation.txt");
	if (!file.is_open()) {
		cout << "File could not be open\n";
		exit(0);
	}
	while (!file.eof()) {
		vector <double> input;
		vector <double> output;
		//bias input
		input.push_back(1);
		double value;
		double dValue;
		//read input and output and normalize them
		for (int j = 1; j<inputNeurons.size(); j++) {
			file >> value;
			value = (value - normalizer[j - 1][0]) / (normalizer[j - 1][1] - normalizer[j - 1][0]);
			input.push_back(value);
		}
		for (int j = 0; j<outputNeurons.size(); j++) {
			file >> dValue;
			dValue = (dValue - normalizer[j+2][0]) / (normalizer[j+2][1] - normalizer[j+2][0]);
			output.push_back(dValue);
		}

		//go through the input neurons	
		runInputs(input);

		//go through the output neurons
		runOutputLayer();

		//calculate errors
		vector <double> error = calculateError(output);

		sumError += error[0]*error[0] + error[1]*error[1];
		count++;
	}
	return sumError / count;
}

/*
This method obtains the minimum and maximum values for each of the
inputs and outputs of the file. This is then used throught the rest
of the routines to normalize input and output. It also inserts the 
values into the vector used to find similar values in online training
*/
vector <vector <double> > NeuralNetwork::normalizeValues(string file) {
	double minL = 1000;
	double maxL = 0;
	double minR = 1000;
	double maxR = 0;
	double minF = 1000;
	double maxF = 0;
	double minRS = 1000;
	double maxRS = 0;
	double value[4];
	ifstream reader;
	reader.open(file);
	while (!reader.eof()) {
		reader >> value[0];
		minF = min(value[0], minF);
		maxF = max(value[0], maxF);
		reader >> value[1];
		minRS = min(value[1], minRS);
		maxRS = max(value[1], maxRS);
		reader >> value[2];
		minL = min(value[2], minL);
		maxL = max(value[2], maxL);
		reader >> value[3];
		minR = min(value[3], minR);
		maxR = max(value[3], maxR);
		originalData.push_back(make_pair(make_pair(value[0], value[1]), make_pair(value[2], value[3])));
	}
	sort(originalData.begin(), originalData.end());
	vector <double> front;
	front.push_back(minF);
	front.push_back(maxF);
	vector <double> rightS;
	rightS.push_back(minRS);
	rightS.push_back(maxRS);
	vector <double> left;
	left.push_back(minL);
	left.push_back(maxL);
	vector <double> right;
	right.push_back(minR);
	right.push_back(maxR);
	vector <vector< double> > res;
	res.push_back(front);
	res.push_back(rightS);
	res.push_back(left);
	res.push_back(right);
	return res;
}
#endif
