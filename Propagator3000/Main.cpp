#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{

	cv::Mat image = cv::imread("input.BMP", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1, 1.0 / 255);

	CpuGator holo1(image), holo2(image), holo3(768, 768);
	float distance = 10e-3; //mm

	holo3.MulTransferFunction(distance);
	holo3.ShowAll();
	
	std::cout << "Dx propagation\n";
	holo1.Sobel('x');
	holo1.Propagate1D(distance, 'x');
	holo1.ShowAll();

	std::cout << "Dy propagation\n";
	holo2.Sobel('y');
	holo2.Propagate1D(distance, 'y');
	holo2.ShowAll();

}