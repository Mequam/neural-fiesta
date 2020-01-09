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
//begin backpropigation tests
	std::cout << std::endl << "[test] BEGINING BACKPROP TESTS!!!" << std::endl; 
	//set up the new network
	cn = NNet::ConvNetwork({1,2,2});
	//run it with the given input value
	cn.run(td2.input_value);
	std::cout << "{{" << cn.LayerList[0][0].activation << "},{" << cn.LayerList[1][0].activation << "," << cn.LayerList[1][1].activation << "}}" << std::endl;
	std::cout << "w0:" << cn.LayerList[1][0].cons[0].weight << " w1:" << cn.LayerList[1][1].cons[0].weight << std::endl;
	//create a list of the derivatives of the output layer functions to that value
	std::vector<double> derivO(cn.LayerList[cn.LayerList.size()-1].size());
	for (int i = 0; i < cn.LayerList[cn.LayerList.size()-1].size(); i++)
	{
		derivO[i] = 2*(cn.LayerList[cn.LayerList.size()-1][i].activation-td2.wanted_output[i]);
	}

	//allocate memory for the weights
	std::vector<double> weight_mem(cn.weight_size());
	std::cout << "[test] initilized " << weight_mem.size() << " weight memory locations" << std::endl;
	int weight_offset = 0; //set the offset into that memory chunk to zero
	
	//allocate memory for the biases
	std::vector<double> bias_mem(cn.neuron_count()-cn.LayerList[0].size());
	std::cout << "[test] initilized " << bias_mem.size() << " bias memory" << std::endl;
	int bias_offset = 0; //set the offset into the biases to zero
	
	//allocate memory for the derivatives of the next layers to be derivatives	
	std::vector<double> derivative_mem(cn.LayerList[1].size());

	//run the backpropigation function for that layer 	
	cn.backprop(cn.LayerList[1],cn.LayerList[2],derivO,&weight_offset,&weight_mem,&bias_offset,&bias_mem,&derivative_mem);
	
	
	for (int i = 0; i < weight_mem.size();i++)
	{
		std::cout << "[test] weight " << i << " had a derivitive of " << weight_mem[i] << std::endl;
	}
	std::cout << "[test] this network contains " << cn.neuron_count() << " neurons" << std::endl;
	for (int i = 0; i < bias_mem.size(); i++)
	{
		std::cout << "[test] bias " << i <<  " had a derivitive of " << bias_mem[i] << std::endl;
	}
	
	for (int i = 0; i < derivative_mem.size();i++)
	{
		std::cout << "[test] the derivatives for the second layer are " << derivative_mem[i] << std::endl;
	}
return 0;
}
