#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{
	cv::Mat image = cv::imread("Images/input.BMP", cv::IMREAD_GRAYSCALE);
	
	image.convertTo(image, CV_32FC1, 1.0 / 255);
	cv::copyMakeBorder(image, image, 1024, 1024, 1024, 1024, cv::BORDER_CONSTANT);
	
	float z2 = 1;
	float focal = 2 * z2;
	CpuGator ai(image), lens = CpuGator::Lens(ai.Data.rows, ai.Data.cols, -focal, ai.PitchX, ai.PitchY, ai.Wavelength);
	ai *= lens;
	ai.ShowAll();
	ai.Propagate(z2); 
	ai.ShowAll();
	ai.Re();
	ai.Propagate(-z2);
	ai.ShowAll();


	system("pause");
	return 0;
}
