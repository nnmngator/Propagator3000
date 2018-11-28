#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{

	cv::Mat image = cv::imread("C:\\CGH\\input_wframe.BMP", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1, 1.0 / 255);
	

	CpuGator holo(image);

	holo.PitchX = 10e-6;
	holo.PitchY = 10e-6;

	/*
	#########################################################
	#####			Calculation of Hologram				#####
	#########################################################
	*/
	
	
	//backward propagation
	holo.Propagate1D(-10e-3,'x');
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);

	holo.IntNormCplx();
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);
	holo.PhaseBinCplx();
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);


	/*
	#########################################################
	#####			Reconstruction of Hologram			#####
	#########################################################
	*/
	holo.Propagate(10e-3);
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::Phase);
	std::cout << holo.PitchX << "reverse propagated form propagate method" << std::endl;

}