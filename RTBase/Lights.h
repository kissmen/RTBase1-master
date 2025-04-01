#pragma once

#include "Core.h"
#include "Geometry.h"
#include "Materials.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

class SceneBounds
{
public:
	Vec3 sceneCentre;
	float sceneRadius;
};

class Light
{
public:
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf) = 0;
	virtual Colour evaluate(const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isArea() = 0;
	virtual Vec3 normal(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float totalIntegratedPower() = 0;
	virtual Vec3 samplePositionFromLight(Sampler* sampler, float& pdf) = 0;
	virtual Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf) = 0;
	BSDF* getBSDF() { return new DiffuseBSDF(); }
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 emittingNormal;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		//if (Dot(wi, triangle->gNormal()) < 0)
		if (Dot(wi.normalize(), emittingNormal) > EPSILON)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Origin area PDF
		float pdfArea = 1.0f / triangle->area;
		// Estimating the location of light source
		Vec3 center = triangle->centre();
		// ShadingData.x to centre square
		float dist2 = (shadingData.x - center).lengthSq();
		// Cosine of direction of emission from the lamp surface with respect to its normal
		//float cosTheta = fabs(Dot(wi, triangle->gNormal()));
		float cosTheta = fabs(Dot(wi, emittingNormal));
		if (cosTheta < 1e-6f) return 0.0f;
		// Area PDF convert to stereo angle PDF
		float pdfSolid = pdfArea * (dist2 / cosTheta);
		return pdfSolid;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		//return triangle->gNormal();
		return emittingNormal;
	}
	float totalIntegratedPower()
	{
		return (triangle->area * emission.Lum());
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		return triangle->sample(sampler, pdf);
	}
	
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Add code to sample a direction from the light
		/*Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);*/
		float r1 = sampler->next();
		float r2 = sampler->next();

		// Hemisphere sample localDir
		Vec3 localDir = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// PDF = cos(theta) / PI
		pdf = SamplingDistributions::cosineHemispherePDF(localDir);

		// Transformation from light normals to world space
		Frame frame;
		//frame.fromVector(-triangle->gNormal());
		//frame.fromVector(triangle->gNormal());
		frame.fromVector(emittingNormal);
		// Returns the direction of the converted world
		return frame.toWorld(localDir);
	}

};

class BackgroundColour : public Light
{
public:
	Colour emission;
	BackgroundColour(Colour _emission)
	{
		emission = _emission;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		reflectedColour = emission;
		return wi;
	}
	Colour evaluate(const Vec3& wi)
	{
		return emission;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return SamplingDistributions::uniformSpherePDF(wi);
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		return emission.Lum() * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 p = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		p = p * use<SceneBounds>().sceneRadius;
		p = p + use<SceneBounds>().sceneCentre;
		pdf = 4 * M_PI * use<SceneBounds>().sceneRadius * use<SceneBounds>().sceneRadius;
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
};

class EnvironmentMap : public Light
{
public:
	Texture* env;
	std::vector<float> marginalDist;
	std::vector<float> conditionalDist;
	float totalEnv;
	int W, H;
	EnvironmentMap(Texture* _env)
	{
		env = _env;
		W = env->width;
		H = env->height;
		buildDistribution();
	}
	void buildDistribution()
	{
		marginalDist.resize(H);
		conditionalDist.resize(W * H);

		std::vector<float> rowSum(H, 0.0f);
		for (int y = 0; y < H; y++)
		{
			// theta
			float theta = M_PI * ((float)y + 0.5f) / (float)H;
			float sinT = sinf(theta);
			for (int x = 0; x < W; x++)
			{
				// Map luminance
				float lum = env->texels[y * W + x].Lum();
				float val = lum * sinT;
				conditionalDist[y * W + x] = val;
				rowSum[y] += val;
			}
		}

		for (int y = 0; y < H; y++)
		{
			float accum = 0.0f;
			for (int x = 0; x < W; x++)
			{
				accum += conditionalDist[y * W + x];
				conditionalDist[y * W + x] = accum;
			}
			// Normalisation
			if (rowSum[y] > 1e-8f)
			{
				for (int x = 0; x < W; x++)
				{
					conditionalDist[y * W + x] /= rowSum[y];
				}
			}
			else
			{
				for (int x = 0; x < W; x++)
					conditionalDist[y * W + x] = 1.0f;
			}
		}
		float accumRow = 0.0f;
		for (int y = 0; y < H; y++)
		{
			accumRow += rowSum[y];
			marginalDist[y] = accumRow;
		}

		totalEnv = accumRow;
		// (0,1)
		if (totalEnv < 1e-8f) totalEnv = 1e-8f;
		for (int y = 0; y < H; y++)
		{
			marginalDist[y] /= totalEnv;
		}
	}
	// Sampling function that samples one direction in environment map based on CDF
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// MarginalDist sampling longitude
		float r1 = sampler->next();
		int row = binarySearchSegment(marginalDist.data(), H, r1);
		// Sampling latitude
		float r2 = sampler->next();
		int col = binarySearchSegment(&conditionalDist[row * W], W, r2);
		// Conversion to spherical coordinate system
		/*float u = ((float)col + 0.5f) / (float)W;
		float v = ((float)row + 0.5f) / (float)H;*/
		float u = (float)col / (float)W;
		float v = (float)row / (float)H;
		float theta = v * M_PI;
		float phi = u * 2.0f * M_PI;

		// Spherical Coordinate Conversion
		float sinT = sinf(theta);
		float cosT = cosf(theta);
		Vec3 dir = Vec3(sinT * cosf(phi), cosT, sinT * sinf(phi));

		// Evaluating colour values
		reflectedColour = evaluate(dir);

		// PDF
		float pixelLum = env->texels[row * W + col].Lum();
		float pixelValue = pixelLum * sinT;
		float localProb = pixelValue / totalEnv;
		pdf = localProb * ((float)W * (float)H / (2.0f * M_PI * M_PI));

		//return shadingData.x + dir * 1e6f;
		// return sampleImportance(sampler, reflectedColour, pdf);
		return dir;
	}

	// Intensity of light source in direction
	Colour evaluate(const Vec3& wi)
	{
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float v = acosf(wi.y) / M_PI;
		return env->sample(u, v);  // Get colour values from the env map using u,v coord
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Assignment: Update this code to return the correct PDF of luminance weighted importance sampling
		float cosT = wi.y;
		if (cosT < -1.f) cosT = -1.f;
		if (cosT > 1.f)  cosT = 1.f;
		float theta = acosf(cosT);
		float phi = atan2f(wi.z, wi.x);
		if (phi < 0) phi += 2.0f * M_PI;

		float v = theta / M_PI;
		float u = phi / (2.0f * M_PI);

		// Related col
		int row = (int)floor(v * (float)H);
		int col = (int)floor(u * (float)W);
		if (row < 0) row = 0;
		if (row >= H) row = H - 1;
		if (col < 0) col = 0;
		if (col >= W) col = W - 1;

		// pixelEnergy = (lum*sinθ)
		float lum = env->texels[row * W + col].Lum();
		float sinT = sinf(theta);
		float pixelValue = lum * sinT;
		float localProb = pixelValue / totalEnv;

		float pdf = localProb * ((float)W * (float)H / (2.0f * M_PI * M_PI));
		return pdf;
	}
	bool isArea()
	{
		return false;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return -wi;
	}
	float totalIntegratedPower()
	{
		float total = 0;
		for (int i = 0; i < env->height; i++)
		{
			float st = sinf(((float)i / (float)env->height) * M_PI);
			for (int n = 0; n < env->width; n++)
			{
				total += (env->texels[(i * env->width) + n].Lum() * st);
			}
		}
		total = total / (float)(env->width * env->height);
		return total * 4.0f * M_PI;
	}
	Vec3 samplePositionFromLight(Sampler* sampler, float& pdf)
	{
		// Samples a point on the bounding sphere of the scene. Feel free to improve this.
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec3 dir = SamplingDistributions::uniformSampleSphere(r1, r2);
		Vec3 p = use<SceneBounds>().sceneCentre + dir * use<SceneBounds>().sceneRadius;
		pdf = 1.f / (4.f * M_PI * SQ(use<SceneBounds>().sceneRadius));
		return p;
	}
	Vec3 sampleDirectionFromLight(Sampler* sampler, float& pdf)
	{
		// Replace this tabulated sampling of environment maps
		Vec3 wi = SamplingDistributions::uniformSampleSphere(sampler->next(), sampler->next());
		pdf = SamplingDistributions::uniformSpherePDF(wi);
		return wi;
	}
	inline int binarySearchSegment(const float* cdf, int size, float r)
	{
		int left = 0;
		int right = size - 1;
		while (left < right)
		{
			int mid = (left + right) >> 1;
			if (cdf[mid] < r)
				left = mid + 1;
			else
				right = mid;
		}
		return left;
	}
	void buildImportanceSampling()
	{
		marginalDist.resize(H);
		std::vector<float> rowSum(H, 0.0f);

		for (int y = 0; y < H; y++)
		{
			float theta = M_PI * ((float)y + 0.5f) / (float)H;
			float sinT = sinf(theta);
			for (int x = 0; x < W; x++)
			{
				// Env map luminance
				float lum = env->texels[y * W + x].Lum();
				rowSum[y] += lum * sinT;
			}
		}

		// CDF
		for (int y = 0; y < H; y++)
		{
			if (rowSum[y] > 1e-8f)
			{
				marginalDist[y] = rowSum[y] / rowSum[H - 1];
			}
			else
			{
				marginalDist[y] = 1.0f;
			}
		}
	}
	Vec3 sampleImportance(Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		// MarginalDist sampling latitude theta
		float r1 = sampler->next();
		int row = binarySearchSegment(marginalDist.data(), H, r1);

		// ConditionalDist sampling longitude phi dir
		float r2 = sampler->next();
		int col = binarySearchSegment(&conditionalDist[row * W], W, r2);

		// Transformation using spherical coordinate system
		float u = (float)col / (float)W;
		float v = (float)row / (float)H;
		float theta = v * M_PI;
		float phi = u * 2.0f * M_PI;

		float sinT = sinf(theta);
		float cosT = cosf(theta);
		Vec3 direction(sinT * cosf(phi), cosT, sinT * sinf(phi));

		// Dir colour
		emittedColour = evaluate(direction);

		// Dir PDF
		float pixelLum = env->texels[row * W + col].Lum();
		float pixelValue = pixelLum * sinT;
		float localProb = pixelValue / totalEnv;
		pdf = localProb * ((float)W * (float)H / (2.0f * M_PI * M_PI));

		return direction;
	}

};