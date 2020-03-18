#include "convNet/convoluted_network.h"
#include <iostream>
#include <vector>
int main()
{
	for (int i = 0; i < 28*28;i++)
	{
		//create a convoluted network in memory that can compute the images
		NNet::ConvNetwork cn({i,4});
		
		NNet::training_data td;
	 	for (int j = 0; j < i;j++)
		{
			td.input_value.push_back(0);
		} 
		td.wanted_output = {0};

		std::cout << "[DEBUG] running backpropigation with " << i << " input neurons" << std::endl;
		cn.full_backprop({td});
	}
	return 0;
}


