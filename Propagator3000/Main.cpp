#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{
	cv::Mat image = cv::imread("Images/lena.BMP", cv::IMREAD_GRAYSCALE);
	
	image.convertTo(image, CV_32FC1, 1.0 / 255);
	
	CpuGator c1(image), c2(image), c3(image);
	c1.Propagate(100e-3, Direction::X);
	c1.Show(FieldType::Intensity);

	c2.Propagate(100e-3, Direction::Y);
	c2.Show(FieldType::Intensity);

	c3.Propagate(100e-3);
	c3.Show(FieldType::Intensity);

	auto c4 = c1 * c2;	 // lub c1 + c2
	c4.Show(FieldType::Intensity);

	system("pause");
	return 0;
}
