#pragma once
#include "GatorUtils.h"
#include<opencv2\opencv.hpp>

// To use FieldType w/o namespace qualifier
using namespace optics;
// To use Direction w/o namespace qualifier
using namespace misc;

class CpuGator
{
	// To refer to pixel type as Pix
	typedef cv::Vec2f Pix;

public:
	CpuGator(int rows, int cols);
	CpuGator(const cv::Mat& image, FieldType fieldType = FieldType::Re);
	CpuGator(const CpuGator& other);
	
	~CpuGator() = default;

	void SetImage(const cv::Mat& image, FieldType fieldType = FieldType::Re);

	void FFT();
	void FFT(Direction axis);
	void IFFT();
	void IFFT(Direction axis);
	void FFTShift();
	
	void NormalizeIntensity();
	void BinarizePhase();

	void Propagate(float distance);
	void Propagate(float distance, Direction axis);
	
	void Show(FieldType fieldType = FieldType::Intensity) const;
	void ShowAll() const; 
	void Save(const std::string & filename, FieldType fieldType) const;
	void ShowSave(const std::string & filename, FieldType fieldType) const;
	void ShowSaveAll(const std::string & filePrefix) const;

	Pix Min() const;
	Pix Max() const;

	void MulLens(float focal); //todo - focalX, focalY

	static CpuGator TransferFunction(int rows, int cols, float distance, float pitchX, float pitchY, float wavelength);
	static CpuGator Lens(int rows, int cols, float focal, float pitchX, float pitchY, float wavelength);
		
	CpuGator& operator=(const CpuGator& other);
	CpuGator& operator=(const Pix& value);

	void operator+=(const CpuGator& other);
	void operator-=(const CpuGator& other);
	void operator*=(const CpuGator& other);
	void operator/=(const CpuGator& other);

	void operator+=(const Pix& value);
	void operator-=(const Pix& value);
	void operator*=(const Pix& value);
	void operator/=(const Pix& value);

	CpuGator& operator+(const CpuGator& other);
	CpuGator& operator-(const CpuGator& other);
	CpuGator& operator*(const CpuGator& other);
	CpuGator& operator/(const CpuGator& other);

	CpuGator& operator+(const Pix& value);
	CpuGator& operator-(const Pix& value);
	CpuGator& operator*(const Pix& value);
	CpuGator& operator/(const Pix& value);
	
public:
	/// <summary>
	/// 2D floating-point data containing complex amplitude:
	/// [Re, Im]
	/// </summary>
	cv::Mat Data;

	/// <summary>
	/// Sampling in X (columns) direction
	/// </summary>
	float PitchX;

	/// <summary>
	/// Sampling in Y (rows) direction
	/// </summary>
	float PitchY;
	
	float Wavelength;

private: 
	void MulTransferFunction(float distance);

	/// <summary>
	/// Sets given pixel value, depending on what field type 
	/// that value corresponds to:
	/// <para>
	/// Re: only Re value is set, leaving Im as it was
	/// </para> 
	/// <para>
	/// Im: only Im value is set, leaving Re as it was
	/// </para> 
	/// <para>
	/// Phase: Re and Im values are set so that amplitude is
	/// 1 and phase is given value
	/// </para>
	/// <para>
	/// Amplitude: Re and Im values are scaled so that phase
	/// remains the same but amplitude is set to given value
	/// </para>
	/// <para>
	/// Intensity: Re and Im values are scaled so that phase
	/// remains the same but intensity (amplitude squared)
	/// is set to given value
	/// </para>
	/// <para>
	/// IntensityLog: Re and Im values are scaled so that phase
	/// remains the same but log of intensity (log of amplitude 
	/// squared) is set to given value
	/// </para>
	/// </summary>
	/// <param name="pix">Pixel to modify</param>
	/// <param name="value">Value to set</param>
	/// <param name="fieldType">Field type value corresponds to</param>
	void SetPixelValue(Pix& pix, float value, FieldType fieldType);

	/// <summary>
	/// Returns pixel value, depending on what field type is
	/// desired.
	/// <para>Re: 1st channel value is returned</para> 
	/// <para>Im: 2nd channel value is returned</para> 
	/// <para>
	/// Phase: Corresponding phase is return as Atan(Im/Re)
	/// </para>
	/// <para>
	/// Amplitude: Corresponding aplitude is returned as Abs(pix)
	/// </para>
	/// <para>Intensity: Amplitude squared is returned</para>
	/// <para> 
	/// IntensityLog: Log of amplitude squared is returned
	/// </para>
	/// </summary>
	/// <param name="pix">Pixel to get value of</param>
	/// <param name="fieldType">Desired field type</param>
	/// <returns></returns>
	float GetPixelValue(const Pix& pix, FieldType fieldType) const;
};

