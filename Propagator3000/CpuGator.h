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

	void FFT();
	void FFT(char axis);
	void IFFT();
	void IFFT(char axis);

	void FFTShift();
	inline void FFTShifted();
	inline void IFFTShifted();
	
	void CplxToExp();
	void ExpToCplx();

	void IntNormExp();
	void IntNormCplx();
	void PhaseBinExp(float threshold = 0);
	void PhaseBinCplx(float threshold = 0);

	//void Resize(int rows, int cols);
	void MulTransferFunction(float distance);
	void MulLens(float focal); //todo - focalX, focalY
	void Propagate(float distance);
	void Propagate1D(float distance, char axis);
	void Show(FieldType fieldType = FieldType::Intensity);

public:
	cv::Mat m_data;
	float PitchX, PitchY, Wavelength;
};

