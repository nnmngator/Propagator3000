#pragma once
#include<opencv2\opencv.hpp>

/*
CpuGator holo(2048,2048);
holo.SetImage(img);
holo.PitchX = 10e-3;
holo.Wavelength = 420e-9;
holo.Propagate(1e-3);
holo.Show(Re);
//~holo();
*/

class CpuGator
{
public:
	enum FieldType {
		Re,
		Im,
		Phase,
		Amplitude,
		Intensity,
		LogIntensity
	};

	CpuGator(int rows, int cols);
	CpuGator(const cv::Mat& image, FieldType fieldType = FieldType::Re);
	~CpuGator() = default;

	void SetImage(const cv::Mat& image, FieldType fieldType = FieldType::Re);

	void FFT2();
	void FFT1rows();
	void FFT1cols();
	void IFFT2();
	void FFTShift();
	inline void FFTShifted() {
		FFTShift();
		FFT2();
		FFTShift();
	}
	inline void IFFTShifted() {
		FFTShift();
		IFFT2();
		FFTShift();
	}
	
	//void Resize(int rows, int cols);
	void MulTransferFunction(float distance);
	void MulLens(float focalX, float focalY);
	void Propagate(float distance);
	void Show(FieldType fieldType = FieldType::Intensity);

public:
	cv::Mat m_data;
	float PitchX, PitchY, Wavelength;
};

