#include <vector>
#include "neuron.h"
#include <iostream>
namespace NNet {
	class ConvNetwork 
	{

		public:
		std::vector<std::vector<neuron>> LayerList;	
		//this initilizer takes a list of comma seperated layer sizes and generates a neuron to match the criteria
		ConvNetwork(std::vector<int>);
	};

	//this initilizer takes a list of comma seperated layer sizes and generates a neuron to match the criteria	 
	ConvNetwork::ConvNetwork(std::vector<int> layer_dims)
	{
		int size = layer_dims.size();
		this->LayerList = std::vector<std::vector<neuron>>(size);
		
		//initilize the first layer of neurons in the convoluted network to have null connections	
		this->LayerList[0] = std::vector<neuron>(layer_dims[0],{1,{}});	
		for (int i = 1; i < size; i++)
		{
			std::cout << layer_dims[i];
			//create a new layer of neurons and add it to the network
			this->LayerList[i] = std::vector<neuron>(layer_dims[i]);
				

			//foreach neuron in the current layer
			for (int j = 0; j < layer_dims[i]; j++)
			{
				//create a list of connections inside of that neuron
				this->LayerList[i][j].cons = std::vector<connection>(layer_dims[i-1]);
				//foreach neuron in the previous layer, add them to the connection with a randomized weight 
				for (int k = 0; k < layer_dims[i-1]; k++)
				{
					this->LayerList[i][j].cons[k] = 
					{
						//address of the neron to connect in the previous layer
						&(this->LayerList[i-1][k]),
						//random value
						//TODO: make it ACTUALY random >_>
						1
					};
				}	
			}	
		}
	}
}

