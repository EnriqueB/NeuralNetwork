#include <iostream>
#include "neuralNetwork.h"

#define TRAININGMEMORY 2000
#define VALIDATIONMEMORY 2000
#define TESTINGMEMORY 1000

#define INPNEURONS 8
#define HIDLAYERS 1
#define HIDNEURONS 20
#define OUTNEURONS 2

int main(int argc, char *argv[]) {
	//Command line arguments are used in order to facilitate
	//the use of the program from the python interface
	if (argc == 1) {
		cout << "Invalid usage.\n" << "Correct usage: \n" << "./network option\n";
		cout << "Valid options: \n" << "A: Train the classifier\n" << "B: Run the classifier over the test file\n" << "C: Predict a crime\n";
		cout << "The first time the program is run, the A parameter must be called first.\n";
		cout << "The program will not run over the test file or predict if it has not been trained first.\n";
		exit(0);
	}
	//The parameter A will train the network
	if (argv[1][0] == 'A') {
		//Dynamic memory is used here to speed up 
		//the training process. Since the data set 
		//is too big, the program would'nt allow
		//for a normal memory allocation
		double** data = new double*[TRAININGMEMORY];
		for (int i = 0; i < TRAININGMEMORY; i++) {
			data[i] = new double[INPNEURONS+OUTNEURONS];
		}
		double** validationData = new double*[VALIDATIONMEMORY];
		for (int i = 0; i < VALIDATIONMEMORY; i++) {
			validationData[i] = new double[INPNEURONS+OUTNEURONS];
		}

		//Constructor parameters: lambda, eta, momentum, input neurons,
		//hidden layers, hidden neurons per layer, output neurons
		NeuralNetwork network(0.6, 0.6, 0.4, INPNEURONS, HIDLAYERS, HIDNEURONS, OUTNEURONS);
		//The data and validation dynamic arrays
		//are passed to the network. The network
		//uses them by reference
		network.run(5000, data, validationData);
		network.saveToFile("saved.txt");

		//Free memory
		for (int i = 0; i < TRAININGMEMORY; i++) {
			delete[] data[i];
		}
		delete[] data;
		for (int i = 0; i < VALIDATIONMEMORY; i++) {
			delete[] validationData[i];
		}
		delete[] validationData;
	}
	else if (argv[1][0] == 'B') {
		double** data = new double*[TESTINGMEMORY];
		for (int i = 0; i < TESTINGMEMORY; i++) {
			data[i] = new double[INPNEURONS+OUTNEURONS];
		}
		NeuralNetwork network("saved.txt");
		network.runTest(data);
		//free memory
		for (int i = 0; i < TESTINGMEMORY; i++) {
			delete[] data[i];
		}
		delete[] data;
	}
	else if (argv[1][0] == 'C') {
		NeuralNetwork network("saved.txt");
		network.predict();
	}
	else {
		cout << "Invalid usage.\n" << "Correct usage: \n" << "./network option\n";
		cout << "Valid options: \n" << "A: Train the classifier\n" << "B: Run the classifier over the test file\n" << "C: Predict a crime\n";
		cout << "The first time the program is run, the A parameter must be called first.\n";
		cout << "The program will not run over the test file or predict a crime if it has not been trained first.\n";
		exit(0);
	}
	cout << "Finished\n";
	return 0;
}
