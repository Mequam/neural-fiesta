#include <vector>
namespace NNet {
	struct neuron;
	struct connection
	{
		neuron* np;
		double weight;
	};
	struct neuron 
	{	
		double activation;
		std::vector<connection> cons;	
	};
}
