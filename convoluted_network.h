#include <vector>
#include "neuron.h"
namespace NNet {
	class ConvNetwork 
	{

		public:
		std::vector<std::vector<neuron>> LayerList;	
		//this initilizer takes a list of comma seperated layer sizes and generates a neuron to match the criteria
		ConvNetwork(std::vector<int>);
		//this function sets the activation for each neuron in the network that is not the starting neurons
		void run(std::vector<double>); //set the initial starting neurons activation to the doubles in this vector 
		void run();//run with the current activations stored in the starting neurons
	};
	void ConvNetwork::run()
	{
		int size = this->LayerList.size();
		for (int i = 1; i < size;i++) //foreach layer
		{
			int size2 = this->LayerList[i].size();
			for (int i2 = 0; i2 < size2; i2++) //foreach neuron in that layer
			{
				//update the value of the neuron
				this->LayerList[i][i2].calc();
			}
		}
	}
	void ConvNetwork::run(std::vector<double> nL)
	{
		//update the start neurons
		int given_size = nL.size();
		int layer_size = this->LayerList[0].size();
		for (int i = 0; i < given_size && i < layer_size;i++)
		{
			this->LayerList[0][i].activation = sigmoid(nL[i]); 
		}

		//update all of the other neurons
		this->run();
	}
	//this initilizer takes a list of comma seperated layer sizes and generates a neuron to match the criteria	 
	ConvNetwork::ConvNetwork(std::vector<int> layer_dims)
	{
		int size = layer_dims.size();
		this->LayerList = std::vector<std::vector<neuron>>(size);
		
		//initilize the first layer of neurons in the convoluted network to have null connections	
		this->LayerList[0] = std::vector<neuron>(layer_dims[0],{1,0,{}});	
		for (int i = 1; i < size; i++)
		{	
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

