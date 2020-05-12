#include "../convNet/convoluted_network.h"
#include <fstream>
#include <string>
#include <stdint.h>
#include <iostream>
#include <vector>

#define IDX_BYTESWAPP

namespace NNet {
	class idx {
		//the actual file connection that we use to read from the db
		std::ifstream infile_img;
		std::ifstream infile_lbl;

		//the magic number of the file that we read, if this is not 2049 there is an issue
		uint32_t magic_num = 0;
		//how many images are stored in the file
		uint32_t img_count = 0;
		//the dimensions of each of those images
		uint32_t rows = 0;
		uint32_t cols = 0;
		uint32_t img_size = 0;
		public:
			idx(std::string,std::string);
			~idx();	
			uint32_t magicNum();
			uint32_t rowSize();
			uint32_t colSize();
			uint32_t imgCount();
			NNet::training_data getTrainingData();

			std::vector<NNet::training_data> getDataSet(int);
			
			void printTd(training_data td);
	};

	//this function returns a vector of training data for the neral net to use to train
	std::vector<NNet::training_data> idx::getDataSet(int size=10) {
		std::vector<NNet::training_data> ret_val;
		for (int i = 0; i < size; i++) {
			ret_val.push_back(getTrainingData());
		}
		return ret_val;
	}	
	//this function outputs a given image to the terminal as ascii artwork for debugging
	void idx::printTd(training_data td) {
		int i = 0;	
		for (std::vector<double>::iterator it = td.wanted_output.begin(); it != td.wanted_output.end();it++,i++) {
			if (*it == 1)
				std::cout << "label: " << i << std::endl;	
		}
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
		uint8_t buff;
		for (int i = 0;i < img_size;i++) {
			infile_img.read((char *)&buff,sizeof(uint8_t));
			ret_val.input_value.push_back(buff);
		}
		infile_lbl.read((char *)&buff,sizeof(uint8_t));

		for (uint8_t i = 0; i < 10; i++) {
			if (i == buff) {
				ret_val.wanted_output.push_back(1);	
			}
			else
				ret_val.wanted_output.push_back(0);
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

	idx::idx(std::string image_file,std::string label_file) {
		infile_img.open(image_file,std::ios::in);
		infile_lbl.open(label_file,std::ios::in);
	
		if (!infile_img.is_open())
			std::cout << "[ERROR] unable to open the image file!" << std::endl;
		if (!infile_lbl.is_open())
			std::cout << "[ERROR] unable to open the label file!" << std::endl;	
		//TODO:add more sanitization to make sure that the file contains the required data 
		infile_lbl.seekg(8);
	
		infile_img.read((char *)&magic_num,sizeof(uint32_t));
		infile_img.read((char *)&img_count,sizeof(uint32_t));	
		infile_img.read((char *)&rows,sizeof(uint32_t));
		infile_img.read((char *)&cols,sizeof(uint32_t));	
		
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
		infile_img.close();
		infile_lbl.close();
	}
}
