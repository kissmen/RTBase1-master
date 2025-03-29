#pragma once

#include "Core.h"
#include <random>
#include <algorithm>

class Sampler
{
public:
	virtual float next() = 0;
};

class MTRandom : public Sampler
{
public:
	std::mt19937 generator;
	std::uniform_real_distribution<float> dist;
	MTRandom(unsigned int seed = 1) : dist(0.0f, 1.0f)
	{
		generator.seed(seed);
	}
	float next()
	{
		return dist(generator);
	}
};


// Note all of these distributions assume z-up coordinate system
class SamplingDistributions
{
public:
	static Vec3 uniformSampleHemisphere(float r1, float r2)
	{
		// r1 is used for cos(theta)
		float cosTheta = 1.0f - r1; // cos(theta) ranges from [0, 1]
		float sinTheta = sqrt(1.0f - cosTheta * cosTheta); // sin(theta)

		// r2 is used for phi (the azimuthal angle)
		float phi = 2.0f * M_PI * r2; // phi ranges from [0, 2pi]

		// Convert to Cartesian coordinates (x, y, z) on the hemisphere
		float x = sinTheta * cos(phi);
		float y = sinTheta * sin(phi);
		float z = cosTheta;

		// Return the vector in world space
		return Vec3(x, y, z);
	}

	static float uniformHemispherePDF(const Vec3 wi)
	{
		if (wi.z <= 0.0f)
			return 0.0f;
		return wi.z / float(M_PI);
	}

	static Vec3 cosineSampleHemisphere(float r1, float r2)
	{
		float cosTheta = sqrt(1.0f - r1);
		float theta = acos(cosTheta);
		float phi = 2.0f * M_PI * r2;

		// call sphericalToWorld(θ, φ)
		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float cosineHemispherePDF(const Vec3 wi)
	{
		// If z<=0，means not on up hemisphere range，pdf=0
		if (wi.z <= 0.0f)
			return 0.0f;

		// wi.z  cos(θ), pdf = cos(θ)/π
		return wi.z / float(M_PI);
	}

	static Vec3 uniformSampleSphere(float r1, float r2)
	{
		// z= cos(θ) range [-1,1]
		//   cosTheta = 1 - 2*r1 => θ = arccos(cosTheta)
		float cosTheta = 1.0f - 2.0f * r1;
		float theta = acos(cosTheta);

		//   φ = 2π * r2
		float phi = 2.0f * M_PI * r2;

		return SphericalCoordinates::sphericalToWorld(theta, phi);
	}

	static float uniformSpherePDF(const Vec3& wi)
	{
		return 1.0f / (4.0f * float(M_PI));
	}
};