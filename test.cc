#include <vector>
#include "convoluted_network.h"
#include <iostream>
#include <string>

int main()
{
 	std::vector<int> args = {200,2,3,4};
	NNet::ConvNetwork cn(args);
	std::cout << cn.LayerList[1].size() << std::endl;	
	if (cn.LayerList[0][0].cons.size() == 0) {
		std::cout << 
		"[main] the first connection is null!" 
			<< std::endl;
	}
	for (int i = 0; i < cn.LayerList[2].size(); i++){
		std::cout << cn.LayerList[2][i].cons.size() << std::endl;
	}
	for (int i = 0; i < cn.LayerList[1].size(); i++){
		std::cout << cn.LayerList[1][i].cons[0].weight << ',' <<
		cn.LayerList[1][i].cons[1].weight << std::endl;
	}
	

	//testing to make sure that the neurons calculate their activation properly
	args = {1,2};
	cn = NNet::ConvNetwork(args);
	cn.LayerList[0][0].activation = 1;
	cn.run(std::vector<double>{0});	
	std::cout << "[test] the activation of the first neuron in the second layer is: " << cn.LayerList[1][0].activation << std::endl;
	std::cout << "[test] the activation of the second neuron in the second layer is: " << cn.LayerList[1][1].activation << std::endl;	
	
	//make sure that the cost function of the network is working properly
	std::cout << "[test] the cost of the network with respect to {.622459,.622459} is " << cn.cost(std::vector<double>{.622459,.622459}) << std::endl;
	
	//make sure that training data is properly handled
	NNet::training_data td1 = {{0},{1,1}};
	NNet::training_data td2 = {{1},{0,0}};
	
	std::cout << "[test] the cost of the network with respect to td {0}->{1,1} is " << cn.cost(td1) << std::endl;
	std::cout << "[test] the cost of the network with respect to td {1}->{0,0} is " << cn.cost(td2) << std::endl;	
	
	std::cout << "[test] the true cost of the network with respect to {0}->{1,1} and {1}->{0,0} is " << cn.trueCost(std::vector<NNet::training_data>{td1,td2}) << std::endl;
	std::cout << "[test] the current network ({1,2}) has " << cn.weight_size() << " weights" << std::endl;
	cn = NNet::ConvNetwork({1,2,3});
	std::cout << "[test] a network of {1,2,3} has " << cn.weight_size() << " weights" << std::endl;
	return 0;
}
