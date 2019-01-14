#pragma once
class GatorComplex
{
public:
	float Re, Im;

	GatorComplex();
	GatorComplex(float re, float im);
	~GatorComplex();

	float Abs() const;
	float Phase() const;
	float Intensity() const;
	GatorComplex Polar() const;


};

