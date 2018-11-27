#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main() 
{

	cv::Mat image = cv::imread("C:\\CGH\\input_wframe.BMP", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1, 1.0 / 255);

	CpuGator holo(image);
	std::cout << holo.PitchX << ", " << holo.PitchY << ", " << holo.Wavelength;

	holo.Propagate(420e-3);
	holo.Show(CpuGator::Re);

	holo.FFTShift();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);

	holo.FFT2();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::LogIntensity);

	holo.FFTShift();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);

	holo.MulTransferFunction(1e-3);
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::LogIntensity);

	holo.FFTShift();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);

	holo.IFFT2();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);
	holo.Show(CpuGator::LogIntensity);

	holo.FFTShift();
	holo.Show(CpuGator::Re);
	holo.Show(CpuGator::Im);




}