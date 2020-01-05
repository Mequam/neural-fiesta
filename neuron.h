#include <vector>
#include <cmath>
namespace NNet {
	double sigmoid(double x){
		//TODO: we need to make a jaged version of this function for speed
		return 1/(1+pow(M_E,-x));
	}	
	struct neuron;
	struct connection
	{
		neuron* np;
		double weight;
	};
	struct neuron 
	{	
		double activation;
		double bias;
		std::vector<connection> cons;
		double getAct(){
			double new_act = bias;
			//we dont want to make a function call every loop iteration
			int size = cons.size(); 
			for (int i = 0; i < size; i++) {
				new_act += this->cons[i].np->activation * this->cons[i].weight;
			}
			return new_act;
		}
		void calc(){
			//simple wrapper function to set the activation of this neuron
			this->activation=sigmoid(this->getAct());
		}	
	};
}
