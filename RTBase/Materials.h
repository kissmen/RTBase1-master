#pragma once
#include <cmath> 
#include <algorithm> 
#include <functional> 

#include "Core.h"
#include "Imaging.h"
#include "Sampling.h"

#pragma warning( disable : 4244)

using namespace std;

class BSDF;

class ShadingData
{
public:
	Vec3 x;
	Vec3 wo;
	Vec3 sNormal;
	Vec3 gNormal;
	float tu;
	float tv;
	Frame frame;
	BSDF* bsdf;
	float t;
	ShadingData() {}
	ShadingData(Vec3 _x, Vec3 n)
	{
		x = _x;
		gNormal = n;
		sNormal = n;
		bsdf = NULL;
	}
};

class ShadingHelper
{
public:
	static float fresnelDielectric(float cosTheta, float iorInt, float iorExt)
	{
		float sinThetaI = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		float sinThetaT = (iorInt / iorExt) * sinThetaI;
		if (sinThetaT >= 1.0f)
			return 1.0f; // 全反射
		float cosThetaT = sqrtf(std::max(0.0f, 1.0f - sinThetaT * sinThetaT));
		float rs = (iorInt * cosTheta - iorExt * cosThetaT) / (iorInt * cosTheta + iorExt * cosThetaT);
		float rp = (iorExt * cosTheta - iorInt * cosThetaT) / (iorExt * cosTheta + iorInt * cosThetaT);
		return 0.5f * (rs * rs + rp * rp);
	}
	static Colour fresnelConductor(float cosTheta, Colour ior, Colour k)
	{
		// Add code here
		 // 计算每个通道的平方值
		float cosTheta2 = cosTheta * cosTheta;
		float sinTheta2 = 1.0f - cosTheta2;
		Colour ior2 = ior * ior;
		Colour k2 = k * k;

		// tmp = ior^2 + k^2
		Colour tmp = ior2 + k2;

		// Rs 分量：
		// Rs = ((ior^2 + k^2) - 2*ior*cosTheta + cosTheta^2) / ((ior^2 + k^2) + 2*ior*cosTheta + cosTheta^2)
		Colour Rs = (tmp - (ior * (2.0f * cosTheta)) + Colour(cosTheta2, cosTheta2, cosTheta2))
			/ (tmp + (ior * (2.0f * cosTheta)) + Colour(cosTheta2, cosTheta2, cosTheta2));

		// Rp 分量：
		// Rp = ((ior^2 + k^2)*cosTheta^2 - 2*ior*cosTheta + 1) / ((ior^2 + k^2)*cosTheta^2 + 2*ior*cosTheta + 1)
		Colour Rp = ((tmp * Colour(cosTheta2, cosTheta2, cosTheta2)) - (ior * (2.0f * cosTheta)) + Colour(1.0f, 1.0f, 1.0f))
			/ ((tmp * Colour(cosTheta2, cosTheta2, cosTheta2)) + (ior * (2.0f * cosTheta)) + Colour(1.0f, 1.0f, 1.0f));

		return (Rs + Rp) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
		// Add code here
		// 计算 tanTheta = sqrt(1-cos^2)/cosTheta（防止除 0）
		float cosTheta = wi.z;
		float sinTheta = sqrtf(std::max(0.0f, 1.0f - cosTheta * cosTheta));
		float tanTheta = sinTheta / std::max(cosTheta, 1e-6f);
		return (-1.0f + sqrtf(1.0f + alpha * alpha * tanTheta * tanTheta)) * 0.5f;
	}
	static float Gggx(Vec3 wi, Vec3 wo, float alpha)
	{
		// Add code here
		float lambda_wi = lambdaGGX(wi, alpha);
		float lambda_wo = lambdaGGX(wo, alpha);
		return 1.0f / (1.0f + lambda_wi + lambda_wo);
	}
	static float Dggx(Vec3 h, float alpha)
	{
		// Add code here
		float cosThetaH = h.z;
		if (cosThetaH <= 0.0f)
			return 0.0f;
		float cosThetaH2 = cosThetaH * cosThetaH;
		float alpha2 = alpha * alpha;
		float denom = cosThetaH2 * (alpha2 - 1.0f) + 1.0f;
		return alpha2 / (M_PI * denom * denom);
	}

	// 反射函数：返回反射向量
	static Vec3 reflect(const Vec3& v, const Vec3& n) {
		return v - 2.0f * Dot(v, n) * n;
	}

	// 折射函数：计算折射向量（eta 为折射率比），返回 true 表示折射成功，否则全内反射
	static bool refract(const Vec3& v, const Vec3& n, float eta, Vec3& refracted) {
		float cosI = -Dot(n, v);
		float sin2T = eta * eta * (1.0f - cosI * cosI);
		if (sin2T > 1.0f) return false; // 全内反射
		float cosT = sqrtf(1.0f - sin2T);
		refracted = eta * v + (eta * cosI - cosT) * n;
		return true;
	}
};

class BSDF
{
public:
	Colour emission;
	// virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) = 0;
	virtual Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& indirect, float& pdf) = 0;
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual float PDF(const ShadingData& shadingData, const Vec3& wi) = 0;
	virtual bool isPureSpecular() = 0;
	virtual bool isTwoSided() = 0;
	bool isLight()
	{
		return emission.Lum() > 0 ? true : false;
	}
	void addLight(Colour _emission)
	{
		emission = _emission;
	}
	Colour emit(const ShadingData& shadingData, const Vec3& wi)
	{
		return emission;
	}
	virtual float mask(const ShadingData& shadingData) = 0;
};


class DiffuseBSDF : public BSDF
{
public:
	Texture* albedo;
	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add correct sampling code here
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add correct PDF code here
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class MirrorBSDF : public BSDF
{
public:
    Texture* albedo;

    MirrorBSDF() = default;
    MirrorBSDF(Texture* _albedo)
    {
        albedo = _albedo;
    }

    Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) override
    {
        // Perfect reflection using the law of reflection
        Vec3 incident = -shadingData.wo;  // Incoming direction (from camera)
        // Compute the perfect reflection direction
        Vec3 r = ShadingHelper::reflect(incident, shadingData.sNormal);
        
        // Since this is a perfect mirror, the PDF is delta distribution (1)
        pdf = 1.0f;
        reflectedColour = albedo->sample(shadingData.tu, shadingData.tv);  // Albedo color for reflection
        return r;  // Return the reflected direction
    }

    Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override
    {
        // For perfect reflection, there is no diffuse reflection
        return Colour(0.0f, 0.0f, 0.0f);
    }

    float PDF(const ShadingData& shadingData, const Vec3& wi) override
    {
        // For perfect mirror reflection, we use delta PDF
        return 0.0f;  // There is no need for a PDF for a perfect mirror (delta distribution)
    }

    bool isPureSpecular() override
    {
        return true;  // Mirror is purely specular (no diffuse reflection)
    }

    bool isTwoSided() override
    {
        return false;  // Mirror is not two-sided (assuming perfect mirror with only one reflective side)
    }

    float mask(const ShadingData& shadingData) override
    {
        return albedo->sampleAlpha(shadingData.tu, shadingData.tv);  // Alpha masking based on the albedo texture
    }
};



class ConductorBSDF : public BSDF
{
public:
	Texture* albedo;
	Colour eta;
	Colour k;
	float alpha;
	ConductorBSDF() = default;
	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
	{
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Conductor sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Conductor PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};


class GGXBSDF : public BSDF {
public:
	float alpha;  // Roughness parameter

	GGXBSDF(float _alpha) : alpha(_alpha) {}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) override {
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = Colour(1.0f, 1.0f, 1.0f) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override {
		Vec3 half_vector = (shadingData.wo + wi).normalize();
		float D = ShadingHelper::Dggx(half_vector, alpha);
		float G = ShadingHelper::Gggx(wi, shadingData.wo, alpha);
		float F = ShadingHelper::fresnelDielectric(Dot(shadingData.wo, half_vector), 1.5f, 1.0f);
		Colour diffuse = Colour(1.0f, 1.0f, 1.0f) / M_PI;
		Colour specular = Colour(D * G * F, D * G * F, D * G * F) / (4.0f * fabsf(Dot(wi, shadingData.sNormal)) * fabsf(Dot(shadingData.wo, shadingData.sNormal)));
		return diffuse + specular;
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi) override {
		Vec3 half_vector = (shadingData.wo + wi).normalize();
		return ShadingHelper::Dggx(half_vector, alpha) * fabsf(Dot(wi, shadingData.sNormal));
	}

	bool isPureSpecular() override { return false; }
	bool isTwoSided() override { return true; }
	float mask(const ShadingData& shadingData) override { return 1.0f; }
};


class GlassBSDF : public BSDF {
public:
	Texture* albedo;  // 可选贴图
	float intIOR;     // 内部折射率
	float extIOR;     // 外部折射率

	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR; // 典型1.5
		extIOR = _extIOR; // 通常1.0(空气)
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler,
		Colour& reflectedColour, float& pdf) override
	{
		Vec3 normal = shadingData.sNormal;
		Vec3 wo = shadingData.wo;  // out-going direction (camera->surface)
		float cosOut = Dot(normal, wo);

		// 1) 判断 inside/outside
		bool inside = (cosOut < 0.0f);
		if (inside) {
			normal = -normal;
			cosOut = -cosOut;
		}

		// 2) 计算折射率比 eta
		float n1 = inside ? intIOR : extIOR;
		float n2 = inside ? extIOR : intIOR;
		float eta = n1 / n2;

		// 3) Fresnel
		float F = ShadingHelper::fresnelDielectric(cosOut, n1, n2);

		// 4) 根据随机数决定反射还是折射
		float r = sampler->next();
		if (r < F) {
			// ---- 反射分支 ----
			pdf = 1.0f;  // delta -> 唯一方向
			// 反射方向
			Vec3 wi = ShadingHelper::reflect(-wo, normal);
			// 反射能量：只乘 Fresnel
			reflectedColour = Colour(1.0f, 1.0f, 1.0f) * F;
			// 也可 × albedo->sample(...) 做颜色
			return wi;
		}
		else {
			// ---- 折射分支 ----
			Vec3 wi;
			if (ShadingHelper::refract(-wo, normal, eta, wi)) {
				// 成功折射
				pdf = 1.0f;
				// 折射能量：乘 (1-F) * eta^2
				float eta2 = (n2 * n2) / (n1 * n1);
				reflectedColour = Colour(1.0f, 1.0f, 1.0f) * (1.0f - F) * eta2;
				return wi;
			}
			else {
				// 全内反射
				pdf = 1.0f;
				Vec3 wiReflect = ShadingHelper::reflect(-wo, normal);
				reflectedColour = Colour(1.0f, 1.0f, 1.0f) * F;
				return wiReflect;
			}
		}
	}

	// evaluate(...) = 0, 因为纯delta
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override {
		return Colour(0.0f, 0.0f, 0.0f);
	}

	// PDF(...) = 0, delta
	float PDF(const ShadingData& shadingData, const Vec3& wi) override {
		return 0.0f;
	}

	bool isPureSpecular() override { return true; }
	bool isTwoSided() override { return true; }
	float mask(const ShadingData& shadingData) override { return 1.0f; }
};





class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float alpha;
	DielectricBSDF() = default;
	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = 1.62142f * sqrtf(roughness);
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with Dielectric sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with Dielectric PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return false;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo;
	float sigma;
	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Replace this with OrenNayar sampling code
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
		pdf = wi.z / M_PI;
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		wi = shadingData.frame.toWorld(wi);
		return wi;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar evaluation code
		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Replace this with OrenNayar PDF
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}
	bool isPureSpecular()
	{
		return false;
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};

//class PlasticBSDF : public BSDF
//{
//public:
//	Texture* albedo;
//	float intIOR;
//	float extIOR;
//	float alpha;
//	PlasticBSDF() = default;
//	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
//	{
//		albedo = _albedo;
//		intIOR = _intIOR;
//		extIOR = _extIOR;
//		alpha = 1.62142f * sqrtf(roughness);
//	}
//	float alphaToPhongExponent()
//	{
//		return (2.0f / SQ(std::max(alpha, 0.001f))) - 2.0f;
//	}
//	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
//	{
//		// Replace this with Plastic sampling code
//		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
//		pdf = wi.z / M_PI;
//		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
//		wi = shadingData.frame.toWorld(wi);
//		return wi;
//	}
//	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
//	{
//		// Replace this with Plastic evaluation code
//		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
//	}
//	float PDF(const ShadingData& shadingData, const Vec3& wi)
//	{
//		// Replace this with Plastic PDF
//		Vec3 wiLocal = shadingData.frame.toLocal(wi);
//		return SamplingDistributions::cosineHemispherePDF(wiLocal);
//	}
//	bool isPureSpecular()
//	{
//		return false;
//	}
//	bool isTwoSided()
//	{
//		return true;
//	}
//	float mask(const ShadingData& shadingData)
//	{
//		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
//	}
//};
class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;  // Surface color of the plastic
	float intIOR;     // Internal IOR (for specular reflection calculation)
	float extIOR;     // External IOR (for refraction calculation)
	float roughness;  // Roughness value to control glossiness (0.0 to 1.0)

	PlasticBSDF() = default;

	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float _roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		roughness = _roughness;
	}

	// Convert roughness to a Phong exponent for specular reflection
	float roughnessToPhongExponent() const
	{
		return (2.0f / std::max(roughness, 0.001f)) - 2.0f;  // Roughness -> Phong exponent mapping
	}

	// Sample the glossy plastic material: handling both diffuse and specular reflection
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& indirect, float& pdf) override
	{
		// Sample the hemisphere for the incoming direction (this will sample the microfacet distribution)
		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next()); // Sample a hemisphere direction
		pdf = wi.z / M_PI;  // Cosine-weighted PDF for the diffuse component

		// Convert the sample from local coordinates to world coordinates
		wi = shadingData.frame.toWorld(wi);

		// If the material is rough, use GGX for specular reflections
		if (roughness > 0.0f)
		{
			// Calculate the half vector
			Vec3 half_vector = (shadingData.wo + wi).normalize();

			// Compute the GGX distribution for the half vector
			float D = ShadingHelper::Dggx(half_vector, roughness);

			// Compute the geometric shadowing term (G)
			float G = ShadingHelper::Gggx(wi, shadingData.wo, roughness);

			// Calculate the indirect lighting contribution from GGX
			indirect = D * G / (4.0f * fabsf(Dot(half_vector, shadingData.sNormal))) * Colour(1.0f, 1.0f, 1.0f); // Indirect light

			pdf = D * fabsf(Dot(wi, shadingData.sNormal)) / (4.0f * fabsf(Dot(wi, shadingData.sNormal)) * fabsf(Dot(shadingData.wo, shadingData.sNormal)));
		}
		else
		{
			indirect = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		}

		return wi;  // Return the reflected direction
	}

	// Evaluate the BSDF for a given incoming ray direction
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override
	{
		// Diffuse component: simple Lambertian reflection
		Colour diffuse = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;

		// If roughness is non-zero, add specular reflection (Phong model)
		if (roughness > 0.0f)
		{
			// Convert the incoming light direction to the half vector
			Vec3 half_vector = (shadingData.wo + wi).normalize();
			float phongExponent = roughnessToPhongExponent();
			float D = ShadingHelper::Dggx(half_vector, phongExponent);  // GGX specular distribution
			float G = ShadingHelper::Gggx(wi, shadingData.wo, phongExponent);  // Shadowing-masking term
			// Specular reflection added to the diffuse term
			return diffuse + (D * G / (4.0f * fabsf(Dot(half_vector, shadingData.sNormal))));
		}

		return diffuse;
	}

	// PDF for the diffuse reflection
	float PDF(const ShadingData& shadingData, const Vec3& wi) override
	{
		// Calculate the half vector
		Vec3 half_vector = (shadingData.wo + wi).normalize();

		// Calculate the microfacet distribution (D) for the half vector
		float D = ShadingHelper::Dggx(half_vector, roughness);

		// Compute the geometric term (G)
		float G = ShadingHelper::Gggx(wi, shadingData.wo, roughness);

		// Return the PDF using the formula for GGX distribution
		return D * fabsf(Dot(wi, shadingData.sNormal)) / (4.0f * fabsf(Dot(wi, shadingData.sNormal)) * fabsf(Dot(shadingData.wo, shadingData.sNormal)));
	}

	bool isPureSpecular() override
	{
		return false;  // It's not purely specular, it has a diffuse component
	}

	bool isTwoSided() override
	{
		return true;  // Plastic materials can be two-sided
	}

	// Transparency (masking) handling
	float mask(const ShadingData& shadingData) override
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};



class LayeredBSDF : public BSDF
{
public:
	BSDF* base;
	Colour sigmaa;
	float thickness;
	float intIOR;
	float extIOR;
	LayeredBSDF() = default;
	LayeredBSDF(BSDF* _base, Colour _sigmaa, float _thickness, float _intIOR, float _extIOR)
	{
		base = _base;
		sigmaa = _sigmaa;
		thickness = _thickness;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		// Add code to include layered sampling
		return base->sample(shadingData, sampler, reflectedColour, pdf);
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code for evaluation of layer
		return base->evaluate(shadingData, wi);
	}
	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		// Add code to include PDF for sampling layered BSDF
		return base->PDF(shadingData, wi);
	}
	bool isPureSpecular()
	{
		return base->isPureSpecular();
	}
	bool isTwoSided()
	{
		return true;
	}
	float mask(const ShadingData& shadingData)
	{
		return base->mask(shadingData);
	}
};