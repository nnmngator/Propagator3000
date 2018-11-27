#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{

	cv::Mat image = cv::imread("Images/lena.png", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1, 1.0 / 255);
	
	/*
	#########################################################
	#####			Calculation of Hologram				#####
	#########################################################
	*/
	CpuGator holo(image);

	for(float z = 0; z < 1000e-3; z += 10e-3)
	{
		holo.Propagate(10e-3);
		holo.Show(CpuGator::Intensity);
	}
	//backward propagation
	holo.Propagate1D(-10e-3,'x');
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::Phase);
	holo.Show(CpuGator::Intensity);

	holo.IntNormCplx();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::Phase);
	holo.Show(CpuGator::Intensity);

	holo.PhaseBinCplx();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::Phase);
	holo.Show(CpuGator::Intensity);

	/*
	#########################################################
	#####			Reconstruction of Hologram			#####
	#########################################################
	*/
	holo.Propagate(10e-3);
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::Phase);
	holo.Show(CpuGator::Intensity);
	std::cout << holo.PitchX << "reverse propagated form propagate method" << std::endl;

}