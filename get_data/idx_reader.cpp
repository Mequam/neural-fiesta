#include "../convNet/convoluted_network.h"
#include <fstream>
#include <string>
#include <stdint.h>

#include <iostream>
#define IDX_BYTESWAPP

namespace NNet {
	class idx {
		//the actual file connection that we use to read from the db
		std::ifstream infile;
		//the magic number of the file that we read, if this is not 2049 there is an issue
		uint32_t magic_num = 0;
		//how many images are stored in the file
		uint32_t img_count = 0;
		//the dimensions of each of those images
		uint32_t rows = 0;
		uint32_t cols = 0;
		uint32_t img_size = 0;
		public:
			idx(std::string s);
			~idx();	
			uint32_t magicNum();
			uint32_t rowSize();
			uint32_t colSize();
			uint32_t imgCount();
			NNet::training_data getTrainingData();
			void printTd(training_data td);
	};
	//this function outputs a given image to the terminal as ascii artwork for debugging
	void idx::printTd(training_data td) {
		for (int i = 0; i < cols; i++) {
			for (int j = 0; j < rows; j++) {
				if ((uint8_t)td.input_value[j+i*rows] > 0)
					std::cout << '@';
				else
					std::cout << '*';
			}
			std::cout << std::endl;
		}
	}
	NNet::training_data idx::getTrainingData() {
		NNet::training_data ret_val;
		char buff;
		for (int i = 0;i < img_size;i++) {
			infile.read(&buff,sizeof(char));
			ret_val.input_value.push_back(buff);
		}
		return ret_val;
	}
	uint32_t idx::magicNum() {
		return magic_num;
	}
	uint32_t idx::rowSize() {
		return rows;
	}
	uint32_t idx::colSize() {
		return cols;
	}
	uint32_t idx::imgCount() {
		return img_count;
	}

	idx::idx(std::string s) {
		infile.open(s,std::ios::in);
		if (infile.is_open())
			std::cout << "[DEBUG] successfully opened the file!" << std::endl;
		
		//read the file data into our object for ease of use
		infile.read((char *)&magic_num,sizeof(uint32_t));
		infile.read((char *)&img_count,sizeof(uint32_t));
		infile.read((char *)&rows,sizeof(uint32_t));
		infile.read((char *)&cols,sizeof(uint32_t));	
		
		#ifdef IDX_BYTESWAPP
			//if we are defined to be in a system that needs the bytes swapped swapp them
			magic_num = __builtin_bswap32(magic_num);
			img_count = __builtin_bswap32(img_count);
			rows = __builtin_bswap32(rows);
			cols = __builtin_bswap32(cols);
		#endif

		img_size = rows*cols;
	}
	idx::~idx() {
		infile.close();
	}
}
