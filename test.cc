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
	cn.LayerList[0][0].activation = 2;
	cn.run(std::vector<double>{.5});	
	
	std::cout << "[main] the activation of the first neuron in the second layer is: " << cn.LayerList[1][0].activation << std::endl;
	return 0;
}
