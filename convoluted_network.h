#include <vector>
#include <iostream>
#include "neuron.h"
namespace NNet {
	struct training_data
	{
		std::vector <double> input_value;
		std::vector <double> wanted_output; 
	};
	class ConvNetwork 
	{
	
		public:	
		std::vector<std::vector<neuron>> LayerList;		
		//this initilizer takes a list of comma seperated layer sizes and generates a neuron to match the criteria
		ConvNetwork(std::vector<int>);
		//this function sets the activation for each neuron in the network that is not the starting neurons
		void run(std::vector<double>); //set the initial starting neurons activation to the doubles in this vector 
		void run();//run with the current activations stored in the starting neurons
		double cost(std::vector<double>); //calculate the cost of the network with the current output neurons FOR A SINGLE VALUE	
		double cost(training_data); //use the training data to set the network before calculating the cost
		double trueCost(std::vector<training_data>); //average the cost of multiple training examples together to create the true cost 
		int weight_size(); //returns how many weights are in the network
		int neuron_count(); //returns a count of how many neurons are in the network		
		
		//this funtion performs backpropigation between two layers in the network 
		void backprop(std::vector<neuron> ls,std::vector<neuron> lf,
			std::vector<double> lf_derivative,int * weight_offset, 
			std::vector<double> * weights,int *bias_offset,std::vector<double> * biases,
			std::vector<double> * next_l_dervative); 
	};
	int ConvNetwork::neuron_count()
	{
		int ret_val = 0;
		int size = this->LayerList.size();
		for (int i = 0; i < size; i++)
		{
			ret_val+=this->LayerList[i].size();
		}
		return ret_val;
	}
	void ConvNetwork::backprop(
	std::vector<neuron> ls,std::vector<neuron> lf,std::vector<double> lf_derivative,
	int * weight_offset, std::vector<double> * weights,
	int *bias_offset,std::vector<double> * biases,
	std::vector<double>* next_l_derivative)
	{
		std::cout << "[convNet] running backpropigation *^*" << std::endl;
		int lf_size = lf.size();
		//store how many weights each entry of the first neurons will have
		int weight_size = ls.size();
		std::cout << "[convNet] the size of the first layer is " << lf.size() << std::endl;
		for (int i = 0; i < lf_size;i++)
		{
			//store the derivative of the sigmoid function for the neuron that we are looking at
			double nf_constants = lf[i].ndsig()*lf_derivative[i];
			(*biases)[i] = nf_constants; //the derivitive of the bias does not care about the weights
			(*bias_offset)++; //any time we set a bias, imidiatly move to the next bias in the list
			for (int j = 0; j < weight_size; j++)
			{	
				std::cout << "[convNet] previous activation: " << lf[i].cons[j].np->activation << std::endl;	
				//add the way that we want to nudge the wieght to the wieght total
				(*weights)[(*weight_offset)] += lf[i].cons[j].np->activation*nf_constants;
				(*weight_offset)++;//any time that we add to how we want a weight moved, incriment our focus to the next weight
			}
		}
		//iterate through the previous list and compute the derivative
		for (int i = 0; i < weight_size; i++)
		{
			double nl_constants = lf[0].cons[i].np->ndsig();
			for (int j = 0; j < lf_size; j++)
			{
				(*next_l_derivative)[i] += lf[j].cons[i].weight*nl_constants*lf_derivative[j];
			}
		}
	}
	int ConvNetwork::weight_size()
	{
		int ret_val = 0;
		//iterate backwards over the layers in the network to more easily align with backpropigation ordering	
		int max = this->LayerList.size()-1;
		for (int i = 0; i < max;i++)
		{
			ret_val+=LayerList[i].size()*LayerList[i+1].size();
		}
		return ret_val;
	}
	double ConvNetwork::trueCost(std::vector<training_data> td)
	{
		int size = td.size();
		double avg = 0;
		for (int i = 0; i < size; i++)
		{
			this->run(td[i].input_value);
			avg += this->cost(td[i].wanted_output);
		}
		return avg/size;
	}
	double ConvNetwork::cost(std::vector<double> goals)
	{
		double sum = 0;
		//TODO: perhaps save this size value somewhere in the object so it does not need to be computed every cost function?
		int last_index = this->LayerList.size()-1;
		int size = this->LayerList[last_index].size();
		for (int i = 0; i < size; i++)
		{
			sum += (this->LayerList[last_index][i].activation-goals[i])*(this->LayerList[last_index][i].activation-goals[i]);
		}
		return sum;
	}
	double ConvNetwork::cost(training_data td)
	{
		this->run(td.input_value);
		return this->cost(td.wanted_output);
	}
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
			this->LayerList[0][i].activation = nL[i]; 
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

