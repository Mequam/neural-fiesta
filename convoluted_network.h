#include <vector> //unsurprisingly for the vector class

//for generating random numbers to initilize the weights
#include <stdlib.h> //srand, rand
#include <time.h> //time

#include <iostream> //for debuging, to be removed when we remove the print statements (although we might use this for file io)
#include "neuron.h"
namespace NNet {
	struct training_data
	{
		std::vector <double> input_value;
		std::vector <double> wanted_output; 
	};
	class ConvNetwork 
	{
	
		int largest_layer;
		int largest_size;
		public:	
		//getters for the above private variables
		int get_largest_size();
		int get_largest_layer();
		//this list stores all of the neurons for the convoluded network object
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
		//this function backpropigates the ENTIRE network
		void full_backprop(std::vector<training_data>); 
	};
//this variable can be got only
	int ConvNetwork::get_largest_layer()
	{
		return this->largest_layer;
	}
	int ConvNetwork::get_largest_size()
	{
		return this->largest_size;
	}
	void ConvNetwork::full_backprop(std::vector<training_data> data_set)
	{
		int data_size = data_set.size();
		int layer_list_size = this->LayerList.size();
		int first_layer_size = this->LayerList[layer_list_size-1].size();
	//allocate memory that will be used to store how the weights need to be changed
		std::vector<double> weight_mem(this->weight_size());
		//this points to where inside of the weight memory that we are located
		int weight_offset = 0;
	//allocate memory that will be used to store how the biases need to be changed
		std::vector<double> bias_mem(this->neuron_count() - this->LayerList[0].size());
		int bias_offset = 0;
		


		//create the first derivatives that will begin the chain of backpropigation
		std::vector<double> derive_zero(this->largest_size);
		
		//initilize the memory that will serve as a buffer for the next memory list
		std::vector<double> derive_mem(this->largest_size);		
		for (int i = 0; i < data_size; i++)
		{
			//foreach training example

			//jump to the beginning of the memory that holds the weights and biases for each training example
			bias_offset = 0;
			weight_offset = 0;
	
			//run the network with the current training data
			this->run(data_set[i].input_value);			
	
			//populate the above list with the initial derivative
			for (int j = 0; j < first_layer_size; j++)
			{
				derive_zero[j] = 2*(data_set[i].wanted_output[j] - this->LayerList[layer_list_size-1][j].activation);
			}
			for (int j = layer_list_size-1;j >= 1;j--)//do not run for the last layer, as they are the input neurons
			{
				
				
				//set up the memory that will be used for the next layer list	
			
				//start at the output layer and work our way to the input layer
				this->backprop(this->LayerList[j-1],this->LayerList[j],derive_zero,&weight_offset,&weight_mem,&bias_offset,&bias_mem,&derive_mem);
				
				
				//set up the first layer derivatives for the next iteration of the loop
				derive_zero=derive_mem;
				
			}
	
		}
	//average all of the training examples together
			
		//we can use the weight offset to avoid computing the weight_mem size again	
		for (int i = 0;i < weight_offset; i++) //for code clarity here weight_offset=weight_mem.size() 
		{
			weight_mem[i]/=data_size;
		}
		//same hack as above
		for (int i = 0; i < bias_offset;i++)
		{
			bias_mem[i]/=data_size;
		}
	//nudge all of the wieghts and biases by the amount that we calculated
		//jump our offsets to the begining again
		weight_offset = 0;
		bias_offset = 0;
		
		//TODO: if ever there was a reason to learn how to run c++ in parallell on a gpu it was this hideous for loop	
		for (int i = layer_list_size-1; i >= 1; i--)
		{
			//foreach non input layer
			for (int j = 0; j < this->LayerList[i].size();j++)
			{
				//foreach neuron in that layer
				
				//update that neurons bias
				this->LayerList[i][j].bias+=bias_mem[bias_offset];
				bias_offset++;
				
				for (int k = 0; k < this->LayerList[i][j].cons.size(); k++)
				{
					//foreach wieght attached to that neuron
					
					//update that wieght
					this->LayerList[i][j].cons[k].weight+=weight_mem[weight_offset];
					weight_offset++;				
				}
			}
		}
	}
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
		std::cout << "last derivative: " << (*next_l_derivative)[0] << std::endl;
		std::cout << "first derivative: " << lf_derivative[0] << std::endl;
		int lf_size = lf.size();
		//store how many weights each entry of the first neurons will have
		int weight_size = ls.size();
		
		for (int i = 0; i < lf_size;i++)
		{
			//store the derivative of the sigmoid function for the neuron that we are looking at
			double nf_constants = lf[i].ndsig()*lf_derivative[i];
	
			(*biases)[*bias_offset] = nf_constants; //the derivitive of the bias does not care about the weights
			(*bias_offset)++; //any time we set a bias, imidiatly move to the next bias in the list
			for (int j = 0; j < weight_size; j++)
			{	
				
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
		
		std::cout << "last derivative for return: " << (*next_l_derivative)[0] << std::endl;
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
		//seed the random number generator
		srand(time(NULL));

		int size = layer_dims.size();
		this->LayerList = std::vector<std::vector<neuron>>(size);
			
		
		//initilize the "king of the hill" that the larger values amoung the neurons will be fighting for
		this->largest_layer = 1;
		this->largest_size = layer_dims[1];

//initilize the first layer of neurons in the convoluted network to have null connections	
		
		this->LayerList[0] = std::vector<neuron>(layer_dims[0],{1,0,{}});	
		for (int i = 1; i < size; i++)
		{	
			//check to see if the given layer is larger than the other layers
			if (layer_dims[i] > layer_dims[this->largest_layer])	
			{
				this->largest_layer = i;
				this->largest_size = layer_dims[i];
			}

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
						(double)(rand()%100)/(double)100 
					};
				}	
			}	
		}
	}
}

