#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <map>
#include <iomanip>
#include <string.h>
#include <stdio.h>
#include "neuron.h"

using namespace std;

class NeuralNetwork {
private:
	//network parameters
	double lambda;
	double eta;
	double momentum;
	double** data;
	double** validationData;
	int ammountValidation;
	int ammountData;

	//network neurons
	vector <Neuron> inputNeurons;
	vector <vector <Neuron> > hiddenLayers;
	vector <Neuron> outputNeurons;

	vector <vector <double> >normalizer;
	int confusionMatrix[2][2];

public:
	NeuralNetwork(double l, double e, double m,
		int inputCount, int layersCount,
		int hiddenCount, int outputCount);
	NeuralNetwork(string fileName);
	void run(int epochs, double** dat, double** val);
	void runInputs(const vector <double> &input);
	void runHiddenLayer();
	void runOutputLayer();
	vector <double> calculateError(const vector <double> &output);
	vector <double> calculateOutputGradients(const vector <double> &error);
	vector <double> calculateInputGradients(const vector <double> &outputGradients);
	void updateOutputWeights(const vector <double> &outputGradients);
	void updateInputWeights(const vector <double> &inputGradients, const vector <double> &input);
	double runValidation(bool calculateMatrix);
	void getNormalizer();
	void saveToFile(string fileName);
	void runTest(double** dat);
	void predict();

};

/*
Constructor. Sets the values and creates the
specified ammount of neurons
*/
NeuralNetwork::NeuralNetwork(double l, double e, double m,
	int inputCount, int layersCount,
	int hiddenCount, int outputCount) {
	lambda = l; eta = e; momentum = m;
	ammountData = 0;
	ammountValidation = 0;

	memset(confusionMatrix, 0, sizeof(confusionMatrix));
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
Constructor. Reads the information from a file
*/
NeuralNetwork::NeuralNetwork(string fileName) {
	inputNeurons.clear();
	outputNeurons.clear();
	for (int i = 0; i < hiddenLayers.size(); i++) {
		hiddenLayers[i].clear();
	}
	ifstream file;
	file.open(fileName.c_str());

	if (!file) {
		cout << "File could not be open. Try training first.\n";
		exit(0);
	}
	//read parameters
	file >> lambda >> eta >> momentum;
	int inputSize, hiddenSize, outputSize;
	vector <int> layerSizes;
	file >> inputSize >> hiddenSize;
	int val;
	for (int i = 0; i < hiddenSize; i++) {
		file >> val;
		layerSizes.push_back(val);
	}
	file >> outputSize;

	//read input neurons
	for (int i = 0; i < inputSize; i++) {
		vector <double> weights(layerSizes[0]);
		for (int j = 0; j <layerSizes[0]; j++) {
			file >> weights[j];
		}
		Neuron n(weights);
		inputNeurons.push_back(n);
	}

	//read hidden neurons
	for (int i = 0; i < hiddenSize - 1; i++) {
		vector <Neuron> neurons;
		for (int j = 0; j < layerSizes[i]; j++) {
			vector <double> weights(layerSizes[i + 1]);
			for (int l = 0; l < layerSizes[i + 1]; l++) {
				file >> weights[l];
			}
			Neuron n(weights);
			neurons.push_back(n);
		}
		hiddenLayers.push_back(neurons);
	}

	//weights of the last layer
	int l = hiddenSize - 1;
	vector <Neuron> neurons;
	for (int i = 0; i < layerSizes[l]; i++) {
		vector <double> weights(outputSize);
		for (int j = 0; j < outputSize; j++) {
			file >> weights[j];
		}
		Neuron n(weights);
		neurons.push_back(n);
	}
	hiddenLayers.push_back(neurons);

	for (int i = 0; i<outputSize; i++) {
		Neuron n;
		outputNeurons.push_back(n);
	}

	//read normalizer
	normalizer.clear();
	vector <double> maxValues;
	vector<double> minValues;
	double value;
	for (int i = 0; i < inputSize + outputSize - 1; i++) {
		file >> value;
		maxValues.push_back(value);
	}
	for (int i = 0; i < inputSize + outputSize - 1; i++) {
		file >> value;
		minValues.push_back(value);
	}
	normalizer.push_back(maxValues);
	normalizer.push_back(minValues);

	file.close();
}

/*
This method runs the network over the training
set for the specified ammount of epochs
*/
void NeuralNetwork::run(int epochs, double** dat, double** val) {
	data = dat;
	validationData = val;
	double sumError;
	getNormalizer();
	cout << "Starting run\n";
	vector <double> error;
	vector <double> outputGradients;
	vector <double> inputGradients;
	double accuracy;

	for (int i = 0; i<epochs; i++) {
		accuracy = 0;
		double avgError = 0;
		double count = 0;
		sumError = 0;
		vector <double> input;
		vector <double> output;
		for (int k = 0; k < ammountData; k++){
			//bias input
			input.push_back(1);
			//read input and output		
			for (int j = 1; j<inputNeurons.size(); j++) {
				input.push_back(data[k][j - 1]);
			}
			for (int j = 0; j<outputNeurons.size(); j++) {
				output.push_back(data[k][j+inputNeurons.size()-1]);
			}

			//go through the input neurons	
			runInputs(input);

			//go through the output neurons
			runOutputLayer();

			//calculate errors
			error = calculateError(output);

			//calculate gradients
			outputGradients = calculateOutputGradients(error);

			//calculate input gradients
			inputGradients = calculateInputGradients(outputGradients);

			//calculate weight updating for the
			//weights going towards output neurons
			updateOutputWeights(outputGradients);

			//calculate weight updating for the
			//weights going towards output neurons
			updateInputWeights(inputGradients, input);


			//calculate the average error and accuracy
			double highest = -10.0;
			int ii = 0;
			int realOne = 0;
			for (int u = 0; u < outputNeurons.size(); u++) {
				avgError += pow(data[k][u + inputNeurons.size() - 1] - outputNeurons[u].getH(), 2.0);
				if (highest < outputNeurons[u].getH()) {
					highest = outputNeurons[u].getH();
					ii = u;
				}
				if (data[k][u + inputNeurons.size() - 1] >= 0.5) {
					realOne = u;
				}
			}


			//accuracy calculation
			if (realOne == ii) {
				accuracy++;
			}

			count++;
			input.clear();
			output.clear();
		}

		cout << "Epoch "<<i<<" error: "<< avgError / count  <<"  "<<"Correct: "<<accuracy<<" accuracy: "<<accuracy/count<<"  ";

		//run validation
		double valError = runValidation(false);

	}
	double acc = runValidation(true);
	FILE *fp = fopen("results_training_features.txt", "w");

	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			fprintf(fp, "%i ", confusionMatrix[i][j]);
		}
		fprintf(fp, "\n");
	}
	//calculate precision and recall for all classes
	//[0] is precision, [1] is recall
	double statistics[2][2];
	memset(statistics, 0, sizeof(statistics));
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			statistics[0][i] += confusionMatrix[i][j];
			statistics[1][i] += confusionMatrix[j][i];
		}
		//avoid division by 0
		if (statistics[0][i] != 0.0) {
			statistics[0][i] = confusionMatrix[i][i] / statistics[0][i];
		}
		else {
			statistics[0][i] = 0.0;
		}
		if (statistics[1][i] != 0.0) {
			statistics[1][i] = confusionMatrix[i][i] / statistics[1][i];
		}
		else {
			statistics[1][i] = 0.0;
		}
		fprintf(fp, "%lf %lf\n", statistics[0][i], statistics[1][i]);
	}
	fprintf(fp, "%lf", acc);
	fclose(fp);
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
			alphaChange = momentum*hiddenLayers[0][k].getDiff(j);
			deltaWeight = eta*outputGradients[j] * hiddenLayers[0][k].getH() + alphaChange;
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
			alphaChange = momentum*inputNeurons[k].getDiff(j - 1);
			deltaWeight = eta*inputGradients[j - 1] * input[k] + alphaChange;
			inputNeurons[k].updateWeight(deltaWeight, j - 1);
		}
	}
}

/*
This method runs the network over the validation data
*/
double NeuralNetwork::runValidation(bool calculateMatrix) {
	double sumError = 0;
	double count = 0;	
	vector <double> error;
	double val[8];
	double accuracy = 0;
	vector <double> input;
	vector <double> output;
	for (int k = 0; k < ammountValidation; k++){
		//bias input
		input.push_back(1);
		//read input and output
		for (int j = 1; j<inputNeurons.size(); j++) {
			input.push_back(validationData[k][j-1]);
		}
		for (int j = 0; j<outputNeurons.size(); j++) {
			output.push_back(validationData[k][j+inputNeurons.size()-1]);
		}

		//go through the input neurons	
		runInputs(input);

		//go through the output neurons
		runOutputLayer();
		
		//accuracy and error calculation
		calculateError(output);
		double highest = -10.0;
		int ii = 0;
		int realOne = 0;
		for (int i = 0; i < outputNeurons.size(); i++) {
			sumError += pow(validationData[k][i+inputNeurons.size()-1] - outputNeurons[i].getH(), 2.0);
			if (highest < outputNeurons[i].getH()){
				highest = outputNeurons[i].getH();
				ii = i;
			}
			if (validationData[k][i + inputNeurons.size() - 1] >= 0.5) {
				realOne = i;
			}
		}
		count++;

		if (realOne == ii) {
			accuracy++;
		}
		if (calculateMatrix) {
			confusionMatrix[realOne][ii]++;
		}

		input.clear();
		output.clear();
	}
	cout << "Validation error: " << sumError/count << " Correct: "<<accuracy<<" accuracy: "<<accuracy/count<<endl;
	if (calculateMatrix) {
		return accuracy / count;
	}
	else {
		return sumError / count;
	}
}

/*
This method runs the trained neural network over
the test data and produces an output file that
can be uploaded to Kaggle
*/
void NeuralNetwork::runTest(double** dat) {
	
	//Data is read first
	data = dat;
	FILE *fp;
	fp = fopen("preprocessed_testing_noDay_noWeek.txt", "r");
	FILE *out;
	out = fopen("results.csv", "w");
	ammountData = 0;
	ammountValidation = 0;
	
	int s = inputNeurons.size() - 1;
	vector <double> maxValues(s, 0);
	vector <double> minValues(s, 5000);

	if (fp == NULL) {
		fprintf(stderr, "Can't open input file in.list!\n");
		exit(1);
	}
	int x = 0;
	while (fscanf(fp, "%lf", &data[x / s][x % s]) != EOF) {
		maxValues[x % s] = max(maxValues[x % s], data[x / s][x%s]);
		minValues[x % s] = min(minValues[x % s], data[x / s][x%s]);
		x++;
	}
	ammountData = x / s;
	x = 0;
	fclose(fp);
	
	//Results are calculated and printed
	vector <double> input;
	fprintf(out, "Id,ARSON,ASSAULT,BAD CHECKS,BRIBERY,BURGLARY,DISORDERLY CONDUCT,DRIVING UNDER THE INFLUENCE,DRUG/NARCOTIC,DRUNKENNESS,EMBEZZLEMENT,EXTORTION,FAMILY OFFENSES,FORGERY/COUNTERFEITING,FRAUD,GAMBLING,KIDNAPPING,LARCENY/THEFT,LIQUOR LAWS,LOITERING,MISSING PERSON,NON-CRIMINAL,OTHER OFFENSES,PORNOGRAPHY/OBSCENE MAT,PROSTITUTION,RECOVERED VEHICLE,ROBBERY,RUNAWAY,SECONDARY CODES,SEX OFFENSES FORCIBLE,SEX OFFENSES NON FORCIBLE,STOLEN PROPERTY,SUICIDE,SUSPICIOUS OCC,TREA,TRESPASS,VANDALISM,VEHICLE THEFT,WARRANTS,WEAPON LAWS\n");

	for (int i = 0; i < ammountData; i++) {
		input.push_back(1);
		//read input and normalize it			
		for (int j = 1; j<inputNeurons.size(); j++) {
			data[i][j-1] = (data[i][j-1] - minValues[j-1]) / (maxValues[j-1] - minValues[j-1]);
			input.push_back(data[i][j - 1]);
		}

		//go through the input neurons	
		runInputs(input);

		//go through the output neurons
		runOutputLayer();

		//calculate percentage of each category
		double sumOutput = 0;
		for (int j = 0; j < outputNeurons.size(); j++) {
			if (outputNeurons[j].getH()>0) {
				sumOutput += outputNeurons[j].getH();
			}
		}
		fprintf(out, "%d,", i);
		for (int j = 0; j < outputNeurons.size()-1; j++) {
			if (outputNeurons[j].getH()>0) {
				fprintf(out, "%lf,", outputNeurons[j].getH() / sumOutput);
			}
			else {
				fprintf(out, "0.0,");
			}
		}
		if (outputNeurons[outputNeurons.size() - 1].getH() > 0) {
			fprintf(out, "%lf\n", outputNeurons[outputNeurons.size() - 1].getH() / sumOutput);
		}
		else {
			fprintf(out, "0.0\n");
		}
		input.clear();
	}
	fclose(out);
}

/*
This method reads the data, normalizes it
and stores it in the dynamic matrices
*/
void NeuralNetwork::getNormalizer() {
	
	//change this one to min int and max int?
	ammountData = 0;
	ammountValidation = 0;
	vector <double> maxValues(inputNeurons.size()+outputNeurons.size()-1, 0);
	vector <double> minValues(inputNeurons.size()+outputNeurons.size()-1, 5000) ;

	//go through all data
	ifstream reader;
	FILE *fp;
	fp = fopen("training_features.txt", "r");
	if (fp == NULL) {
		fprintf(stderr, "Can't open input file in.list!\n");
		exit(1);
	}
	int x = 0;
	int s = inputNeurons.size() + outputNeurons.size() - 1;
	while (fscanf(fp, "%lf", &data[x / s][x % s]) != EOF) {
		maxValues[x % s] = max(maxValues[x % s], data[x / s][x%s]);
		minValues[x % s] = min(minValues[x % s], data[x / s][x%s]);
		x++;
	}
	ammountData = x / s;
	x = 0;
	/*
	for (int i = 0; i < ammountData; i++) {
		for (int j = 0; j < s; j++) {
			data[i][j] = (data[i][j] - minValues[j]) / (maxValues[j] - minValues[j]);
		}
	}
	*/
	fclose(fp);
	fp = fopen("validation_features.txt", "r");

	//validation data is normalized using training data
	while (fscanf(fp, "%lf", &validationData[x / s][x % s]) != EOF) {
		//validationData[x / s][x%s] = (validationData[x / s][x%s] - minValues[x%s]) / (maxValues[x%s] - minValues[x%s]);
		x++;
	}
	ammountValidation = x / s;
	fclose(fp);
	cout << "Finished normalizing\n";

	cout << endl;
	normalizer.push_back(maxValues);
	normalizer.push_back(minValues);
}

/*
This method saves the relevant information to a file
*/
void NeuralNetwork::saveToFile(string fileName) {
	ofstream file;
	file.open(fileName.c_str());
	//write general information of the network
	file << lambda << " " << eta << " " << momentum<<endl;
	file << inputNeurons.size() << " " << hiddenLayers.size()<<" ";
	for(int i = 0; i < hiddenLayers.size(); i++) {
		file << hiddenLayers[i].size() << " ";
	}
	file << outputNeurons.size() << endl;
	file << setprecision(15);

	//write the weights
	for (int i = 0; i < inputNeurons.size(); i++) {
		for (int j = 0; j < hiddenLayers[0].size(); j++) {
			file << inputNeurons[i].getWeight(j)<<" ";
		}
		file << endl;
	}

	//weights of hidden neurons
	for (int i = 0; i < hiddenLayers.size()-1; i++) {
		for (int j = 0;  j < hiddenLayers[i].size(); j++) {
			for (int l = 0; l < hiddenLayers[i + 1].size(); l++) {
				file << hiddenLayers[i][j].getWeight(l) << " ";
			}
			file << endl;
		}
	}

	//weights of the last layer
	int l = hiddenLayers.size() - 1;
	for (int i = 0; i < hiddenLayers[l].size(); i++) {
		for (int j = 0; j < outputNeurons.size(); j++) {
			file << hiddenLayers[l][i].getWeight(j) << " ";
		}
		file << endl;
	}
	//save normalizer
	for (int i = 0; i < normalizer[0].size(); i++) {
		file << normalizer[0][i] << " ";
	}
	file << endl;
	for (int i = 0; i < normalizer[1].size(); i++) {
		file << normalizer[1][i] << " ";
	}
	file.close();
}

/*
This method saves the prediction 
to a query in a file
*/
void NeuralNetwork::predict() {
	//read information from file
	FILE *fp;
	fp = fopen("predict.csv", "r");

	FILE *out;
	out = fopen("resultsPredict.csv", "w");
	vector <double> input(inputNeurons.size(), 0.0);
	input[0] = 1.0;
	for (int i = 1; i < inputNeurons.size(); i++) {
		fscanf(fp, "%lf", &input[i]);
		input[i] = (input[i] - normalizer[1][i]) / (normalizer[0][i] - normalizer[1][i]);
	}
	runInputs(input);

	runOutputLayer();

	double sumOutput = 0;
	for (int j = 0; j < outputNeurons.size(); j++) {
		if (outputNeurons[j].getH()>0) {
			sumOutput += outputNeurons[j].getH();
		}
	}
	for (int j = 0; j < outputNeurons.size() - 1; j++) {
		if (outputNeurons[j].getH()>0) {
			fprintf(out, "%lf,", outputNeurons[j].getH() / sumOutput);
		}
		else {
			fprintf(out, "0.0,");
		}
	}
	if (outputNeurons[outputNeurons.size() - 1].getH() > 0) {
		fprintf(out, "%lf\n", outputNeurons[outputNeurons.size() - 1].getH() / sumOutput);
	}
	else {
		fprintf(out, "0.0\n");
	}
	fclose(out);
	fclose(fp);

}


#endif