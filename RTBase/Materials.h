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
		float cosTheta2 = cosTheta * cosTheta;
		float sinTheta2 = 1.0f - cosTheta2;
		Colour ior2 = ior * ior;
		Colour k2 = k * k;

		Colour tmp = ior2 + k2;

		Colour Rs = (tmp - (ior * (2.0f * cosTheta)) + Colour(cosTheta2, cosTheta2, cosTheta2))
			/ (tmp + (ior * (2.0f * cosTheta)) + Colour(cosTheta2, cosTheta2, cosTheta2));

		Colour Rp = ((tmp * Colour(cosTheta2, cosTheta2, cosTheta2)) - (ior * (2.0f * cosTheta)) + Colour(1.0f, 1.0f, 1.0f))
			/ ((tmp * Colour(cosTheta2, cosTheta2, cosTheta2)) + (ior * (2.0f * cosTheta)) + Colour(1.0f, 1.0f, 1.0f));

		return (Rs + Rp) * 0.5f;
	}
	static float lambdaGGX(Vec3 wi, float alpha)
	{
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

	static Vec3 reflect(const Vec3& v, const Vec3& n) {
		return v - 2.0f * Dot(v, n) * n;
	}

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
	virtual Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const = 0;
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
	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& reflectedColour,
		float& pdf) override
	{
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);

		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * (1.0f / M_PI);

		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		return wiWorld;
	}

	Colour evaluate(const ShadingData& shadingData,
		const Vec3& wi) const
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (wiLocal.z <= 0.0f)
			return Colour(0.0f, 0.0f, 0.0f);

		return albedo->sample(shadingData.tu, shadingData.tv) * (1.0f / M_PI);
	}

	float PDF(const ShadingData& shadingData,
		const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (wiLocal.z <= 0.0f)
			return 0.0f;

		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	bool isPureSpecular() override { return false; }
	bool isTwoSided()     override { return true; }

	float mask(const ShadingData& shadingData) override
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

    Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
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

    Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const
    {
        // For perfect reflection, there is no diffuse reflection
        return Colour(0.0f, 0.0f, 0.0f);
    }

    float PDF(const ShadingData& shadingData, const Vec3& wi)
    {
        // For perfect mirror reflection, we use delta PDF
        return 0.0f;  // There is no need for a PDF for a perfect mirror (delta distribution)
    }

    bool isPureSpecular()
    {
        return true;  // Mirror is purely specular (no diffuse reflection)
    }

    bool isTwoSided()
    {
        return false;  // Mirror is not two-sided (assuming perfect mirror with only one reflective side)
    }

    float mask(const ShadingData& shadingData)
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

	Vec3 sampleGGX_H(float r1, float r2, float alpha)
	{
		float phi = 2.0f * M_PI * r1;
		float tanTheta2 = (alpha * alpha * r2) / (1.0f - r2 + 1e-6f);
		float theta = atanf(sqrtf(tanTheta2));
		float cosTheta = cosf(theta);
		float sinTheta = sinf(theta);
		return Vec3(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);
	}

	inline Vec3 reflect(const Vec3& I, const Vec3& N)
	{
		return I - N * (2.0f * I.dot(N));
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
	{
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		Vec3 woWorld = shadingData.wo;
		Vec3 woLocal = shadingData.frame.toLocal(woWorld);

		if (woLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}

		float r1 = sampler->next();
		float r2 = sampler->next();
		
		Vec3 whLocal = sampleGGX_H(r1, r2, alpha);
		whLocal.normalize();

		Vec3 wiLocal = reflect(-woLocal, whLocal);
		if (wiLocal.z <= 0.0f)
		{
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return Vec3(0.0f, 0.0f, 0.0f);
		}
		wiLocal.normalize();
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);

		float Dval = ShadingHelper::Dggx(whLocal, alpha);
		float cosWO_WH = fabsf(woLocal.dot(whLocal));
		float cosThetaO = woLocal.z;
		float G = ShadingHelper::Gggx(woLocal, wiLocal, alpha);

		pdf = (Dval * fabsf(whLocal.z)) / (4.0f * fabsf(woLocal.dot(whLocal)) + 1e-6f);


		float cosThetaI = wiLocal.z;

		float cosDH = fabsf(wiLocal.dot(whLocal));
		Colour F = ShadingHelper::fresnelConductor(cosDH, eta, k);

		Colour specular = F * baseColor * (Dval * G / (4.0f * fabsf(woLocal.z) * fabsf(wiLocal.z) + 1e-6f));
		reflectedColour = specular;
		return wiWorld;
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wiWorld) const
	{
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wiWorld);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f)
			return Colour(0.0f, 0.0f, 0.0f);
		Vec3 whLocal = (woLocal + wiLocal).normalize();
		if (whLocal.z < 0.0f)
			whLocal = -whLocal;
		float Dval = ShadingHelper::Dggx(whLocal, alpha);
		float G = ShadingHelper::Gggx(woLocal, wiLocal, alpha);
		float cosThetaO = fabsf(woLocal.z);
		float cosThetaI = fabsf(wiLocal.z);
		float cosDH = fabsf(woLocal.dot(whLocal));
		Colour F = ShadingHelper::fresnelConductor(cosDH, eta, k);
		Colour specular = F * baseColor * (Dval * G / (4.0f * cosThetaO * cosThetaI + 1e-6f));
		return specular;
	}

	float PDF(const ShadingData& shadingData, const Vec3& wiWorld)
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wiWorld);
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f)
			return 0.0f;
		Vec3 whLocal = (woLocal + wiLocal).normalize();
		if (whLocal.z < 0)
			whLocal = -whLocal;
		float Dval = ShadingHelper::Dggx(whLocal, alpha);
		float cosThetaH = fabsf(whLocal.z);
		float pdfSpec = (Dval * cosThetaH) / (4.0f * fabsf(wiLocal.dot(whLocal)) + 1e-6f);
		return pdfSpec;
	}

	bool isPureSpecular() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};



class GlassBSDF : public BSDF {
public:
	Texture* albedo;
	float intIOR;
	float extIOR;

	GlassBSDF(Texture* _albedo, float _intIOR, float _extIOR)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
	}

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler,
		Colour& reflectedColour, float& pdf)
	{
		Vec3 normal = shadingData.sNormal;
		Vec3 wo = shadingData.wo;  // out-going direction (camera->surface)
		float cosOut = Dot(normal, wo);

		// inside/outside
		bool inside = (cosOut < 0.0f);
		if (inside) {
			normal = -normal;
			cosOut = -cosOut;
		}

		// eta
		float n1 = inside ? intIOR : extIOR;
		float n2 = inside ? extIOR : intIOR;
		float eta = n1 / n2;

		// Fresnel
		float F = ShadingHelper::fresnelDielectric(cosOut, n1, n2);

		float r = sampler->next();
		if (r < F) {
			// reflect
			pdf = 1.0f;  // delta
			// reflect dir
			Vec3 wi = ShadingHelper::reflect(-wo, normal);
			reflectedColour = Colour(1.0f, 1.0f, 1.0f) * F;
			return wi;
		}
		else {
			// refract
			Vec3 wi;
			if (ShadingHelper::refract(-wo, normal, eta, wi)) {
				// refract
				pdf = 1.0f;
				float eta2 = (n2 * n2) / (n1 * n1);
				reflectedColour = Colour(1.0f, 1.0f, 1.0f) * (1.0f - F) * eta2;
				return wi;
			}
			else {
				// TIR
				pdf = 1.0f;
				Vec3 wiReflect = ShadingHelper::reflect(-wo, normal);
				reflectedColour = Colour(1.0f, 1.0f, 1.0f) * F;
				return wiReflect;
			}
		}
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const
	{
		return Colour(0.0f, 0.0f, 0.0f);
	}

	// PDF(...) = 0, delta
	float PDF(const ShadingData& shadingData, const Vec3& wi) {
		return 0.0f;
	}

	bool isPureSpecular() { return true; }
	bool isTwoSided() { return false; }
	float mask(const ShadingData& shadingData) { return albedo->sampleAlpha(shadingData.tu, shadingData.tv); }
};


class DielectricBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR; 
	float alpha;

	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = std::max(0.001f, roughness);
	}

	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& outColor,
		float& outPdf) override
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		bool isFront = (woLocal.z > 0.0f);
		float eta = isFront ? (extIOR / intIOR) : (intIOR / extIOR);
		// flip normal if inside
		if (!isFront) woLocal.z = -woLocal.z;

		float r1 = sampler->next();
		float r2 = sampler->next();

		float cosThetaM = std::sqrt((1.0f - r1)
			/ (r1 * (alpha * alpha - 1.0f) + 1.0f));
		float sinThetaM = std::sqrt(std::max(0.f, 1.0f - cosThetaM * cosThetaM));
		float phiM = 2.0f * M_PI * r2;
		Vec3 hLocal(sinThetaM * std::cos(phiM),
			sinThetaM * std::sin(phiM),
			cosThetaM);

		if (Dot(woLocal, hLocal) < 0.0f)
			hLocal = -hLocal;

		// Fresnel
		float cosOH = Dot(woLocal, hLocal);
		float F = ShadingHelper::fresnelDielectric(cosOH, isFront ? intIOR : extIOR, isFront ? extIOR : intIOR);

		float xi = sampler->next();
		bool reflectBranch = (xi < F);

		Vec3 wiLocal;
		if (reflectBranch)
		{
			wiLocal = -woLocal + 2.f * cosOH * hLocal;
			if (wiLocal.z <= 0.f) {
				outPdf = 0.f;
				outColor = Colour(0.f, 0.f, 0.f);
				return shadingData.frame.toWorld(wiLocal);
			}
		}
		else
		{
			
			float cosIH = 0.f;
			Vec3 refr;
			if (!ShadingHelper::refract(-woLocal, hLocal, eta, refr)) {
				
				wiLocal = -woLocal + 2.f * cosOH * hLocal;
				if (wiLocal.z <= 0.f) {
					outPdf = 0.f;
					outColor = Colour(0.f, 0.f, 0.f);
					return shadingData.frame.toWorld(wiLocal);
				}
				reflectBranch = true; // fallback
			}
			else {
				wiLocal = refr;
				if (std::fabs(wiLocal.z) < 1e-6f) {
					outPdf = 0.f;
					outColor = Colour(0.f, 0.f, 0.f);
					return shadingData.frame.toWorld(wiLocal);
				}
			}
		}

		float dotWH = Dot(woLocal, hLocal);
		// D
		float Dval = ShadingHelper::Dggx(hLocal, alpha);
		float Jh = 1.0f / (4.f * dotWH);
		float pdfH = Dval * hLocal.z;
		float pdfVal = pdfH * Jh;

		outPdf = pdfVal;

		if (reflectBranch) {
			outColor = Colour(F,F,F);
		}
		else {
			float ratio = (1.f - F);
			outColor = Colour(ratio, ratio, ratio);
		}

		if (!isFront) {
			wiLocal.z = -wiLocal.z;
		}

		return shadingData.frame.toWorld(wiLocal);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const
	{
		return Colour(0.f, 0.f, 0.f);
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		return 0.f;
	}

	bool isPureSpecular() { return true; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& /*sd*/) { return 1.0f; }
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

	Vec3 sample(const ShadingData& shadingData, Sampler* sampler,
		Colour& reflectedColour, float& pdf)
	{
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(),
			sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);


		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;

		return shadingData.frame.toWorld(wiLocal);
	}


	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const
	{

		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f)
			return Colour(0.0f, 0.0f, 0.0f);

		float theta_i = acosf(std::max(0.0f, wiLocal.z));
		float theta_o = acosf(std::max(0.0f, woLocal.z));
		float phi_i = atan2f(wiLocal.y, wiLocal.x);
		float phi_o = atan2f(woLocal.y, woLocal.x);

		float sigma2 = sigma * sigma;
		float A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
		float B = 0.45f * sigma2 / (sigma2 + 0.09f);

		float cosDeltaPhi = cosf(phi_i - phi_o);
		float alpha = std::max(theta_i, theta_o);
		float beta = std::min(theta_i, theta_o);

		float term = std::max(0.0f, cosDeltaPhi) * sinf(alpha) * tanf(beta);

		float orenNayarValue = A + B * term;

		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		return (baseColor / M_PI) * orenNayarValue;
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi)
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	bool isPureSpecular() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData)
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};



class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;
	float intIOR;
	float extIOR;
	float roughness;

	PlasticBSDF() = default;
	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR, float _roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		roughness = _roughness;
	}

	float roughnessToExponent() const
	{
		return std::max((1.0f - roughness) * 200.0f, 1.0f);
	}

	float fresnelSchlick(float cosTheta, float intIOR, float extIOR) const {
		// 计算 R0
		float eta = extIOR / intIOR;
		float R0 = powf((eta - 1.0f) / (eta + 1.0f), 2.0f);

		cosTheta = fabsf(cosTheta);
		return R0 + (1.0f - R0) * powf(1.0f - cosTheta, 5.0f);
	}

	static inline Vec3 Reflect(const Vec3& I, const Vec3& N)
	{
		float dotVal = I.dot(N);
		return Vec3(I.x - 2.0f * dotVal * N.x,
			I.y - 2.0f * dotVal * N.y,
			I.z - 2.0f * dotVal * N.z);
	}

	// Phong
	Vec3 samplePhongLobe(const Vec3& rLocal, float n_phong, float r1, float r2)
	{
		float theta = acosf(powf(1.0f - r2, 1.0f / (n_phong + 1.0f)));
		float phi = 2.0f * M_PI * r1;
		float sinTheta = sinf(theta);
		float cosTheta = cosf(theta);
		Vec3 L_local(sinTheta * cosf(phi), sinTheta * sinf(phi), cosTheta);

		Vec3 tangent, bitangent;
		buildBasis(rLocal, tangent, bitangent);
		Vec3 wi = tangent * L_local.x + bitangent * L_local.y + rLocal * L_local.z;
		return wi.normalize();
	}

	// sample() - phong lobe
	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& reflectedColour,
		float& pdf)
	{
		Vec3 woWorld = shadingData.wo;
		Vec3 woLocal = shadingData.frame.toLocal(woWorld);
		if (woLocal.z <= 0.f) {
			pdf = 0.f;
			reflectedColour = Colour(0, 0, 0);
			return Vec3(0, 0, 0);
		}
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);

		float eta = extIOR / intIOR;
		float R0 = powf((eta - 1.f) / (eta + 1.f), 2.0f);
		float F = fresnelSchlick(fabsf(woLocal.z), intIOR, extIOR);

		float choice = sampler->next();
		Vec3 wiLocal(0, 0, 0);
		float branchPdf = 0.f;
		Colour branchBRDF(0, 0, 0);
		float n_phong = roughnessToExponent();

		if (choice < F) {
			Vec3 rLocal = Reflect(-woLocal, Vec3(0, 0, 1));
			rLocal.normalize();
			float r1_sample = sampler->next();
			float r2_sample = sampler->next();
			wiLocal = samplePhongLobe(rLocal, n_phong, r1_sample, r2_sample);
			branchPdf = ((n_phong + 1.0f) / (2.0f * M_PI)) * powf(fabsf(rLocal.dot(wiLocal)), n_phong);
			float specVal = ((n_phong + 2.0f) / (2.0f * M_PI)) * powf(fabsf(rLocal.dot(wiLocal)), n_phong);
			branchBRDF = Colour(specVal, specVal, specVal) * F;
		}
		else {
			float r1_sample = sampler->next();
			float r2_sample = sampler->next();
			wiLocal = SamplingDistributions::cosineSampleHemisphere(r1_sample, r2_sample);
			branchPdf = (wiLocal.z / M_PI);
			branchBRDF = baseColor / M_PI;
		}

		float finalPdf = (choice < F) ? (F * branchPdf) : ((1.0f - F) * branchPdf);
		if (finalPdf < 1e-10f) {
			pdf = 0.f;
			reflectedColour = Colour(0, 0, 0);
			return shadingData.frame.toWorld(wiLocal);
		}
		pdf = finalPdf;

		Colour fVal = Colour((1.0f - F), (1.0f - F), (1.0f - F)) * (baseColor / M_PI) + branchBRDF;
		reflectedColour = fVal;
		return shadingData.frame.toWorld(wiLocal);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wiWorld) const
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wiWorld);
		if (woLocal.z <= 0.f || wiLocal.z <= 0.f)
			return Colour(0, 0, 0);

		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		Colour diff = baseColor / M_PI;

		float n_phong = roughnessToExponent();
		Vec3 rLocal = Reflect(-woLocal, Vec3(0, 0, 1));
		rLocal.normalize();
		float specFactor = powf(fmax(rLocal.dot(wiLocal), 0.0f), n_phong);
		float specVal = ((n_phong + 2.0f) / (2.0f * M_PI)) * specFactor;
		float eta = extIOR / intIOR;
		float F = fresnelSchlick(fabsf(woLocal.z), intIOR, extIOR);
		Colour spec = Colour(specVal, specVal, specVal) * F;

		return Colour((1.0f - F), (1.0f - F), (1.0f - F)) * diff + spec;
	}

	float PDF(const ShadingData& shadingData, const Vec3& wiWorld) {
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wiWorld);
		float pdfDiff = (wiLocal.z > 0 ? wiLocal.z / M_PI : 0);
		Vec3 rLocal = Reflect(-woLocal, Vec3(0, 0, 1));
		rLocal.normalize();
		float n_phong = roughnessToExponent();
		float pdfSpec = ((n_phong + 1.0f) / (2.0f * M_PI)) * powf(fabsf(rLocal.dot(wiLocal)), n_phong) / (4.0f * fabsf(woLocal.dot(rLocal)) + 1e-6f);
		float R0 = powf((extIOR - intIOR) / (extIOR + intIOR), 2.0f);
		float F = fresnelSchlick(fabsf(woLocal.z), intIOR, extIOR);
		return F * pdfSpec + (1.0f - F) * pdfDiff;
	}

	bool isPureSpecular() { return false; }
	bool isTwoSided() { return true; }
	float mask(const ShadingData& shadingData) {
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
	static void buildBasis(const Vec3& normal, Vec3& tangent, Vec3& bitangent)
	{
		if (fabsf(normal.x) > 0.9f) {
			tangent = Vec3(0.0f, 1.0f, 0.0f);
		}
		else {
			tangent = Vec3(1.0f, 0.0f, 0.0f);
		}
		bitangent = normal.cross(tangent);

		tangent = tangent.normalize();
		bitangent = bitangent.normalize();

		tangent = bitangent.cross(normal).normalize();
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
		Vec3 wiWorld = base->sample(shadingData, sampler, reflectedColour, pdf);

		if (pdf < 1e-10f || reflectedColour.Lum() < 1e-10f)
			return wiWorld;

		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wiWorld);

		bool transmitted = (woLocal.z * wiLocal.z < 0.0f);

		if (transmitted)
		{
			Colour attenuation = sigmaa.exp(-(sigmaa * thickness));
			reflectedColour = reflectedColour * attenuation;
		}

		return wiWorld;
	}
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) const
	{
		Colour baseVal = base->evaluate(shadingData, wi);

		if (baseVal.Lum() < 1e-10f)
			return baseVal;

		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi); 

		bool transmitted = (woLocal.z * wiLocal.z < 0.0f);

		if (transmitted)
		{
			Colour attenuation = sigmaa.exp(-(sigmaa * thickness));

			baseVal = baseVal * attenuation;
		}

		return baseVal;
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


