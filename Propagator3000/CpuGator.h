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
	typedef uchar field_type;

public:

	enum FieldType : field_type
	{
		Re = 0x01,
		Im = 0x02,
		Phase = 0x04,
		Amplitude = 0x08,
		Intensity = 0x10,
		LogIntensity = 0x20
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
	void FFTShifted();
	void FFTShifted(char axis);
	void IFFTShifted();
	void IFFTShifted(char axis);
	void CplxToExp();
	void ExpToCplx();

	void IntensityNorm();
	void PhaseBinarization(float threshold);

	void Sobel(char axis, int kernelSize = 5);

	//void Resize(int rows, int cols);
	void MulTransferFunction(float distance);
	void MulLens(float focal); //todo - focalX, focalY
	void Propagate(float distance);
	void Propagate1D(float distance, char axis);
	
	void Show(FieldType fieldType = FieldType::Intensity);
	void ShowAll();
	void Save(const std::string & filename, FieldType fieldType);
	void ShowSave(const std::string & filename, FieldType fieldType);
	void ShowSaveAll(const std::string & filePrefix);


public:
	cv::Mat m_data;
	float PitchX, PitchY, Wavelength;

private:

	enum InternalFieldType : field_type
	{
		ReIm = Re | Im, 
		AmpPhase = Amplitude | Phase | Intensity | LogIntensity
	};
	// Holds info whether currently m_data is in [Re, Im] or [Amp, Phase] form
	InternalFieldType m_fieldType;

	void SetPixelValue(cv::Vec2f& pix, float value, FieldType fieldType);
	float GetPixelValue(const cv::Vec2f& pix, FieldType fieldType);
	
	void ToggleFieldType();
	void ToggleFieldType(field_type desiredFieldType);
};

