#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{

	cv::Mat image = cv::imread("C:\\CGH\\input_wframe.BMP", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1, 1.0 / 255);
	
	/*
	#########################################################
	#####			Calculation of Hologram				#####
	#########################################################
	*/
	CpuGator holo(image);
	holo.ShowSaveAll("C:/Export/1input");
	holo.Sobel('x');
	holo.ShowSaveAll("C:/Export/2sobel");

/*
	for(float z = 0; z < 1000e-3; z += 10e-3)
	{
		holo.Propagate(10e-3);
		holo.Show(CpuGator::Intensity);
	}

	*/
	//backward propagation
	holo.Propagate1D(-10e-3,'x');
	holo.ShowSaveAll("C:/Export/3BackProp");
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);
	//holo.Show(CpuGator::Intensity);

	//holo.IntNormCplx();
	//holo.ShowSaveAll("C:/Export/4IntNorm");
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);
	//holo.Show(CpuGator::Intensity);

	//holo.PhaseBinCplx();
	//holo.ShowSaveAll("C:/Export/5PhaseBin");
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);
	//holo.Show(CpuGator::Intensity);

	/*
	#########################################################
	#####			Reconstruction of Hologram			#####
	#########################################################
	*/
	holo.Propagate(10e-3);
	holo.ShowSaveAll("C:/Export/6ForwardProp");
	//holo.Show(CpuGator::Re);
	//holo.Show(CpuGator::Im);
	//holo.Show(CpuGator::Phase);
	//holo.Show(CpuGator::Intensity);
	//std::cout << holo.PitchX << "reverse propagated form propagate method" << std::endl;

}