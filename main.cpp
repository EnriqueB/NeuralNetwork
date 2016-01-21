#include "Aria.h"
#include <iostream>
#include <Windows.h>
#include <time.h>
#include <cstdlib>
#include <string>
#include "neuralNetwork.h"


using namespace std;

void beepTones(ArRobot *robot, const vector< pair<double, unsigned char> >& tune){
	char buf[40];
	size_t p = 0;
	for (std::vector< pair<double, unsigned char> >::const_iterator i = tune.begin(); i != tune.end() && p < 39; ++i)
	{
		if ((*i).first > 0)
		{
			buf[p++] = (char)((*i).first / 20.0);
			buf[p++] = (char)(*i).second;
		}
	}
	buf[p++] = 0;
	buf[p++] = 0;
	robot->comStrN(ArCommands::SAY, buf, p);
}


int main(int argc, char **argv){
	
	Aria::init();
	ArRobot robot;
	ArPose pose;

	ArArgumentParser argParser(&argc, argv);
	argParser.loadDefaultArguments();
	argParser.addDefaultArgument("?connectLaser");


	ArRobotConnector robotConnector(&argParser, &robot);
	if (robotConnector.connectRobot())
		cout << "Robot connected!" << endl;

	robot.runAsync(false);
	robot.lock();
	robot.enableMotors();
	robot.unlock();
	
	
	ArLaserConnector laserConnector(&argParser, &robot,
		&robotConnector);
	if (laserConnector.connectLasers())
		std::cout << "Laser connected!" << std::endl;

	ArLaser *laser = robot.findLaser(1);
	
	

	ArSensorReading *sonarSensor[8];
	
	srand(time(NULL));
	vector < pair < double, unsigned char > >tune(19);
	robot.setVel2(0, 0);
	tune[0] = make_pair(300, 62);
	tune[1] = make_pair(300, 71);
	tune[2] = make_pair(300, 69);
	tune[3] = make_pair(300, 67);

	tune[4] = make_pair(900, 62);
	tune[5] = make_pair(300, 62);

	tune[6] = make_pair(300, 62);
	tune[7] = make_pair(300, 71);
	tune[8] = make_pair(300, 69);
	tune[9] = make_pair(300, 67);

	tune[10] = make_pair(900, 64);
	tune[11] = make_pair(300, 64);

	tune[12] = make_pair(300, 64);
	tune[13] = make_pair(300, 60);
	tune[14] = make_pair(300, 71);
	tune[15] = make_pair(300, 69);
	tune[16] = make_pair(900, 75);
	tune[17] = make_pair(300, 75);
	tune[18] = make_pair(300, 62);
	beepTones(&robot, tune);
	NeuralNetwork network(.6, .8, .6, 2, 1, 2, 2);
	network.run(150);
	beepTones(&robot, tune);
	while (true){
		cout << "Commencing online training. The robot will train until the 'esc' key\n is pressed. After that it will ask if it should train once more\n";
		//online training
		double laserRange[18];
		double laserAngle[18];
		string answer;
		do{
			vector<double> error(2);
			vector <double> laserSensor(3);
			double count = 0;
			while (true){

				if (GetAsyncKeyState(VK_ESCAPE))
					break;

				laser->lockDevice();
				laserSensor[0] = laser->currentReadingPolar(70, 90, &laserAngle[0]);
				laserSensor[1] = laser->currentReadingPolar(20, 45, &laserAngle[0]);
				laser->unlockDevice();
				
				

				vector<vector<double> > outputs = network.onlineTraining(laserSensor[1], laserSensor[0]); 
				error[0] += outputs[1][0]*outputs[1][0];
				error[1] += outputs[1][1]*outputs[1][1];
				//set speed
				robot.setVel2(outputs[0][0], outputs[0][1]);
				count++;
			}
			robot.setVel2(0, 0);
			ofstream outFile;
			outFile.open("errorOnlineTraining.txt");
			outFile <<(error[0]+error[1])/count << endl;
			outFile.close();
			//update weights
			error[0] = pow((error[0] / count),0.5);
			error[1] = pow((error[1] / count),0.5);

			vector <double> outputGradients = network.calculateOutputGradients(error);
			vector <double> inputGradients = network.calculateInputGradients(outputGradients);
			network.updateOutputWeights(outputGradients);
			network.updateInputWeights(inputGradients, laserSensor);


			cout << "Do another round of training? (y/n) ";
			cin >> answer;
		} while (answer == "y");
		break;
		//i++;
		//ArUtil::sleep(10);
	}
	robot.lock();
	robot.stop();
	robot.unlock();
	Aria::exit();
}
