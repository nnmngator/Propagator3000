#include<opencv2\opencv.hpp>
#include "CpuGator.h"

int main()
{
	/*cv::Mat image = cv::imread("Images/input.BMP", cv::IMREAD_GRAYSCALE);

	image.convertTo(image, CV_32FC1, 1.0 / 255);
	cv::copyMakeBorder(image, image, 1024, 1024, 1024, 1024, cv::BORDER_CONSTANT);
	*/
	/*float z2 = 1;
	float focal = 2 * z2;
	CpuGator ai(image), lens = CpuGator::Lens(ai.Data.rows, ai.Data.cols, -focal, ai.PitchXSrc, ai.PitchYSrc, ai.Wavelength);
	ai *= lens;
	ai.ShowAll();
	ai.Propagate(z2);
	ai.ShowAll();
	ai.Re();
	ai.Propagate(-z2);
	ai.ShowAll();*/
	//
	/*CpuGator chirpU(1024, 1024);
	chirpU.Wavelength = 633e-6;
	chirpU.PitchXSrc = chirpU.PitchYSrc = 10e-3;
	chirpU.PitchXDst = chirpU.PitchYDst = 6e-3;
	chirpU.ChirpC(100, 0, 0);
	chirpU.Show(Re);
	chirpU.ChirpU(100, 0, 0);
	chirpU.Show(Re);
	chirpU.ChirpH(100, 0, 0);
	chirpU.Show(Re);*/
	//chirpU.ChirpH(30, 0, 0);
	/*chirpU.ShowAll();
	chirpU.Show(Re);*/

	//CpuGator zjeb(1024, 1024);
	//zjeb.Wavelength = 633e-6;
	//zjeb.PitchXSrc = zjeb.PitchYSrc = 10e-3;
	//zjeb.PitchXDst = zjeb.PitchYDst = 6e-3;

	//zjeb.ChirpH(100,0,0);
	//zjeb.Show(Re);
	//CpuGator cwel(1024,1024);
	//cwel.MulChirpH(100, 0, 0, 1024, 1024, 633e-6, 10e-3, 10e-3, 6e-3, 6e-3);
	//cwel.Show(Re);



	cv::Mat image = cv::imread("Images/lena.bmp", cv::IMREAD_GRAYSCALE);
	image.convertTo(image, CV_32FC1,1.0/255);
	//cv::copyMakeBorder(image, image, 512, 512, 512, 512, cv::BORDER_CONSTANT, cv::Scalar(0));
	CpuGator lol(image);

	float distance = 0.1;
	float offsetX = 0;
	float offsetY = 0;
	float scale = 6.f / 10.f;

	lol.Wavelength = 633e-9;
	lol.PitchXDst =  10e-6;
	lol.PitchYDst = 10e-6;
	lol.PitchXSrc = lol.PitchXDst * scale;
	lol.PitchYSrc = lol.PitchYDst * scale;


/*	lol.ChirpH(distance, offsetX, offsetY);
	lol.ShowAll();

	lol.ChirpU(distance, offsetX, offsetY);
	lol.ShowAll();
	
	lol.ChirpC(distance, offsetX, offsetY);
	lol.ShowAll();*/

	lol.SetImage(image);
	/*lol.FFTShift();
	lol.FFT();
	lol.FFTShift();*/

	lol.ARSSPropagate(distance, offsetX, offsetY);
	lol.Show(Re);
	lol.Show(Intensity);
	lol.Save("pajac", Re);
	
	system("pause");
	return 0;
}
