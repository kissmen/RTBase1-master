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
};

class AreaLight : public Light
{
public:
	Triangle* triangle = NULL;
	Colour emission;
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		emittedColour = emission;
		return triangle->sample(sampler, pdf);
	}
	Colour evaluate(const Vec3& wi)
	{
		if (Dot(wi, triangle->gNormal()) < 0)
		{
			return emission;
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 1.0f / triangle->area;
	}
	bool isArea()
	{
		return true;
	}
	Vec3 normal(const ShadingData& shadingData, const Vec3& wi)
	{
		return triangle->gNormal();
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
		Vec3 wi = Vec3(0, 0, 1);
		pdf = 1.0f;
		Frame frame;
		frame.fromVector(triangle->gNormal());
		return frame.toWorld(wi);
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
	std::vector<float> marginalDist;   // 大小H，每行的累积分布
	std::vector<float> conditionalDist; // 大小W*H，每个像素的累积分布
	float totalEnv;                     // 用于存储贴图总能量 (积分)
	int W, H;                           // 缓存贴图分辨率
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

		// 1) 对每个像素计算 "pixelEnergy = lum * sinTheta"
		//    rowSum[y] 累加每行总和
		std::vector<float> rowSum(H, 0.0f);
		for (int y = 0; y < H; y++)
		{
			// 计算此行中心对应 theta
			float theta = M_PI * ((float)y + 0.5f) / (float)H;
			float sinT = sinf(theta);
			for (int x = 0; x < W; x++)
			{
				// 贴图亮度 (r+g+b 或其他加权)
				float lum = env->texels[y * W + x].Lum();
				float val = lum * sinT;
				conditionalDist[y * W + x] = val;
				rowSum[y] += val;
			}
		}

		// 2) 对每行做前缀和 => conditionalDist[y*W + x] 变成 0..1 CDF
		for (int y = 0; y < H; y++)
		{
			float accum = 0.0f;
			for (int x = 0; x < W; x++)
			{
				accum += conditionalDist[y * W + x];
				conditionalDist[y * W + x] = accum; // 累加
			}
			// 现在 accum == rowSum[y]
			// 归一化
			if (rowSum[y] > 1e-8f)
			{
				for (int x = 0; x < W; x++)
				{
					conditionalDist[y * W + x] /= rowSum[y];
				}
			}
			else
			{
				// 若行能量近0，则整行cdf直接置为1.0
				for (int x = 0; x < W; x++)
					conditionalDist[y * W + x] = 1.0f;
			}
		}

		// 3) 对行Sum做前缀和 => marginalDist[y]
		float accumRow = 0.0f;
		for (int y = 0; y < H; y++)
		{
			accumRow += rowSum[y];
			marginalDist[y] = accumRow;
		}

		totalEnv = accumRow; // 记录总和
		// 归一化 [0..1]
		if (totalEnv < 1e-8f) totalEnv = 1e-8f; // 避免除0
		for (int y = 0; y < H; y++)
		{
			marginalDist[y] /= totalEnv;
		}
	}
	// 采样函数，基于累积分布函数（CDF）采样环境贴图中的一个方向
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// 1) 从 marginalDist 中采样经度（θ方向）
		float r1 = sampler->next();
		int row = binarySearchSegment(marginalDist.data(), H, r1);

		// 计算该行的CDF范围
		float cdfBelow = (row == 0) ? 0.0f : marginalDist[row - 1];
		float rowCDFRange = marginalDist[row] - cdfBelow;
		float rowCDFLocal = (r1 - cdfBelow) / std::max(rowCDFRange, 1e-8f);

		// 2) 在该行中，采样纬度（φ方向）
		float r2 = sampler->next();
		int col = binarySearchSegment(&conditionalDist[row * W], W, r2);

		// 3) 转换为球面坐标系
		float u = ((float)col + 0.5f) / (float)W;
		float v = ((float)row + 0.5f) / (float)H;
		float theta = v * M_PI;
		float phi = u * 2.0f * M_PI;

		// 球面坐标转换
		float sinT = sinf(theta);
		float cosT = cosf(theta);
		Vec3 dir = Vec3(sinT * cosf(phi), cosT, sinT * sinf(phi));

		// 4) 评估颜色值
		reflectedColour = evaluate(dir);

		// 5) 计算该方向的PDF
		float pixelLum = env->texels[row * W + col].Lum();
		float pixelValue = pixelLum * sinT;
		float localProb = pixelValue / totalEnv;
		pdf = localProb * ((float)W * (float)H / (2.0f * M_PI * M_PI));

		//return shadingData.x + dir * 1e6f;
		return sampleImportance(sampler, reflectedColour, pdf);
	}

	// 评估该方向的光源强度
	Colour evaluate(const Vec3& wi)
	{
		float u = atan2f(wi.z, wi.x);
		u = (u < 0.0f) ? u + (2.0f * M_PI) : u;
		u = u / (2.0f * M_PI);
		float v = acosf(wi.y) / M_PI;
		return env->sample(u, v);  // 使用u,v坐标从环境贴图中获取颜色值
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

		float v = theta / M_PI;        // [0..1]
		float u = phi / (2.0f * M_PI); // [0..1]

		// 找对应行 col
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

		// 同样要乘 (W*H)/(2π²)
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
		// 任选场景 bounds
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec3 dir = SamplingDistributions::uniformSampleSphere(r1, r2);
		Vec3 p = use<SceneBounds>().sceneCentre + dir * use<SceneBounds>().sceneRadius;
		// pdf = 1 / (4π R^2) 之类
		// ...
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
		// 假设 cdf[size-1] = 1.0 (或近似)
		// 用标准二分
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
		// 计算边际分布（对于纬度）
		marginalDist.resize(H);
		std::vector<float> rowSum(H, 0.0f);

		for (int y = 0; y < H; y++)
		{
			float theta = M_PI * ((float)y + 0.5f) / (float)H;
			float sinT = sinf(theta);
			for (int x = 0; x < W; x++)
			{
				// 获取环境贴图的亮度
				float lum = env->texels[y * W + x].Lum();
				rowSum[y] += lum * sinT;  // 累加每行的亮度与sinθ的乘积
			}
		}

		// 计算累积分布函数（CDF）
		for (int y = 0; y < H; y++)
		{
			if (rowSum[y] > 1e-8f)
			{
				marginalDist[y] = rowSum[y] / rowSum[H - 1]; // 归一化
			}
			else
			{
				marginalDist[y] = 1.0f;
			}
		}
	}
	Vec3 sampleImportance(Sampler* sampler, Colour& emittedColour, float& pdf)
	{
		// 1) 从边际分布（marginalDist）中采样纬度（theta方向）
		float r1 = sampler->next();
		int row = binarySearchSegment(marginalDist.data(), H, r1);

		// 2) 从条件分布（conditionalDist）中采样经度（phi方向）
		float r2 = sampler->next();
		int col = binarySearchSegment(&conditionalDist[row * W], W, r2);

		// 3) 使用球面坐标系转换
		float u = (float)col / (float)W;
		float v = (float)row / (float)H;
		float theta = v * M_PI;
		float phi = u * 2.0f * M_PI;

		float sinT = sinf(theta);
		float cosT = cosf(theta);
		Vec3 direction(sinT * cosf(phi), cosT, sinT * sinf(phi));

		// 4) 评估该方向的颜色
		emittedColour = evaluate(direction);

		// 5) 计算该方向的PDF
		float pixelLum = env->texels[row * W + col].Lum();
		float pixelValue = pixelLum * sinT;
		float localProb = pixelValue / totalEnv;
		pdf = localProb * ((float)W * (float)H / (2.0f * M_PI * M_PI));

		// 返回该方向
		return direction;
	}

};