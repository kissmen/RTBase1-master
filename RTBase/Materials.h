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
	Texture* albedo;  // 用于存储漫反射颜色贴图（或纯色）

	DiffuseBSDF() = default;
	DiffuseBSDF(Texture* _albedo)
	{
		albedo = _albedo;
	}

	//==================== 1) Sample ====================
	// 用余弦加权半球采样来获得散射方向 (wi)
	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& reflectedColour,
		float& pdf) override
	{
		// (a) 先在局部坐标系中做余弦加权半球采样
		float r1 = sampler->next();
		float r2 = sampler->next();
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);

		// (b) PDF = cosTheta / π (即 wiLocal.z / π)
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);

		// (c) 计算散射系数(简单Lambert) = albedo / π
		//    其中 albedo->sample(...) 返回纹理贴图采样颜色
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) * (1.0f / M_PI);

		// (d) 将采样到的方向从局部坐标系转换回世界坐标系
		Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
		return wiWorld;
	}

	//==================== 2) Evaluate ==================
	// 计算 BSDF 值 f_r(x, wi, wo) = albedo/π (若 wi 在表面上半球)
	Colour evaluate(const ShadingData& shadingData,
		const Vec3& wi) override
	{
		// (a) 将入射方向转换到局部坐标
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		// (b) 若入射方向在下半球 (wiLocal.z <= 0)，则无贡献
		if (wiLocal.z <= 0.0f)
			return Colour(0.0f, 0.0f, 0.0f);

		// (c) Lambert: 返回 albedo / π
		return albedo->sample(shadingData.tu, shadingData.tv) * (1.0f / M_PI);
	}

	//==================== 3) PDF =====================
	// 与余弦加权采样一致
	float PDF(const ShadingData& shadingData,
		const Vec3& wi) override
	{
		// (a) 转到局部
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		// (b) 若在下半球 => pdf=0
		if (wiLocal.z <= 0.0f)
			return 0.0f;

		// (c) 余弦半球 pdf:  wiLocal.z / π
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	//==================== 4) 其他接口 =====================
	bool isPureSpecular() override { return false; }
	bool isTwoSided()     override { return true; }

	float mask(const ShadingData& shadingData) override
	{
		// 若有alpha贴图，则采样，否则返回1
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

class ConductorBSDF : public BSDF {
public:
	Texture* albedo;
	Colour eta, k;  // 金属折射率和吸收系数
	float alpha;    // 粗糙度

	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness) {
		albedo = _albedo;
		eta = _eta;
		k = _k;
		alpha = std::max(0.001f, roughness); // clamp
	}

	//----------- sample(...) -----------
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf) override {
		// 1) 将 outgoing direction (camera -> surface) 转换到局部坐标
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);

		// 2) 采样半向量 h
		float r1 = sampler->next();
		float r2 = sampler->next();
		float theta = atanf(alpha * sqrtf(r1 / (1.0f - r1)));
		float phi = 2.0f * M_PI * r2;
		float cosT = cosf(theta);
		float sinT = sinf(theta);
		Vec3 hLocal(sinT * cosf(phi), sinT * sinf(phi), cosT);

		// 3) 计算反射方向 wiLocal
		float dotWH = Dot(woLocal, hLocal);
		if (dotWH < 0.0f) {
			hLocal = -hLocal;
			dotWH = -dotWH;
		}
		Vec3 wiLocal = -woLocal + 2.0f * dotWH * hLocal;

		// 4) 如果反射到地面下方 (wiLocal.z <= 0), 则采样失败
		if (wiLocal.z <= 0.0f) {
			pdf = 0.0f;
			reflectedColour = Colour(0.0f, 0.0f, 0.0f);
			return shadingData.frame.toWorld(wiLocal);
		}

		// 5) 计算 PDF
		float DVal = ShadingHelper::Dggx(hLocal, alpha); // GGX微表面分布
		float pdfHalf = DVal * hLocal.z / (4.0f * dotWH);
		pdf = pdfHalf;

		// 6) 计算 Fresnel (使用金属的 Fresnel 而非 Dielectric)
		float cosDH = Dot(wiLocal, hLocal);
		Colour F = ShadingHelper::fresnelConductor(cosDH, eta, k);

		reflectedColour = F;
		return shadingData.frame.toWorld(wiLocal);  // 转换回世界坐标
	}

	//----------- evaluate(...) -----------
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override {
		// 1) 转换到局部坐标
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (woLocal.z <= 0 || wiLocal.z <= 0) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		// 2) 计算半向量 h
		Vec3 hLocal = (wiLocal + woLocal).normalize();
		float D = ShadingHelper::Dggx(hLocal, alpha);  // 微表面分布
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alpha);  // 几何遮蔽
		float cosDH = Dot(wiLocal, hLocal);
		Colour F = ShadingHelper::fresnelConductor(cosDH, eta, k);  // Fresnel项

		// 3) 计算微表面反射项
		float denom = 4.0f * std::fabs(wiLocal.z * woLocal.z);
		Colour spec = D * G * F / denom;

		return spec;
	}

	//----------- PDF(...) -----------
	float PDF(const ShadingData& shadingData, const Vec3& wi) override {
		// 与 sample 中计算的 pdf 相同
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		if (woLocal.z <= 0 || wiLocal.z <= 0) {
			return 0.0f;
		}

		Vec3 hLocal = (wiLocal + woLocal).normalize();
		float dotWH = Dot(woLocal, hLocal);
		if (dotWH <= 0) return 0.0f;

		float DVal = ShadingHelper::Dggx(hLocal, alpha);
		float pdfHalf = DVal * hLocal.z / (4.0f * dotWH);
		return pdfHalf;
	}

	bool isPureSpecular() override { return false; }
	bool isTwoSided() override { return false; }
	float mask(const ShadingData& shadingData) override { return 1.0f; }
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
	Texture* albedo;   // 可选贴图用于调节颜色
	float intIOR;      // 内部折射率，如1.5
	float extIOR;      // 外部折射率，如1.0
	float alpha;       // 粗糙度 (GGX展开用)

	DielectricBSDF(Texture* _albedo, float _intIOR, float _extIOR, float roughness)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		alpha = std::max(0.001f, roughness);
	}

	// ---------- sample ----------
	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& outColor,
		float& outPdf) override
	{
		// 1) 将 outgoing direction 转到局部坐标
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		bool isFront = (woLocal.z > 0.0f);
		float eta = isFront ? (extIOR / intIOR) : (intIOR / extIOR);
		// flip normal if inside
		if (!isFront) woLocal.z = -woLocal.z;

		// 2) 先采样 GGX 半向量 hLocal
		float r1 = sampler->next();
		float r2 = sampler->next();

		// 根据课件中 GGX 采样公式：theta_m, phi_m
		float cosThetaM = std::sqrt((1.0f - r1)
			/ (r1 * (alpha * alpha - 1.0f) + 1.0f));
		float sinThetaM = std::sqrt(std::max(0.f, 1.0f - cosThetaM * cosThetaM));
		float phiM = 2.0f * M_PI * r2;
		Vec3 hLocal(sinThetaM * std::cos(phiM),
			sinThetaM * std::sin(phiM),
			cosThetaM);

		// 让hLocal与woLocal同侧(避免dot(woLocal,hLocal)<0)
		if (Dot(woLocal, hLocal) < 0.0f)
			hLocal = -hLocal;

		// 3) Fresnel
		float cosOH = Dot(woLocal, hLocal);
		float F = ShadingHelper::fresnelDielectric(cosOH, isFront ? intIOR : extIOR, isFront ? extIOR : intIOR);

		// 4) 根据F判断反射/折射
		float xi = sampler->next();
		bool reflectBranch = (xi < F);

		Vec3 wiLocal;
		if (reflectBranch)
		{
			// ---- Reflection ----
			wiLocal = -woLocal + 2.f * cosOH * hLocal;
			// 若反射到下半球 => pdf=0, 采样失败
			if (wiLocal.z <= 0.f) {
				outPdf = 0.f;
				outColor = Colour(0.f, 0.f, 0.f);
				return shadingData.frame.toWorld(wiLocal);
			}
		}
		else
		{
			// ---- Refraction ----
			// 参考Walter Microfacet refraction 
			// 这里做简化: wi = refract(-wo, h, eta'), 同时要注意Jacobian 
			float cosIH = 0.f;
			Vec3 refr;
			if (!ShadingHelper::refract(-woLocal, hLocal, eta, refr)) {
				// 全内反射
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

		// 5) 计算pdf
		float dotWH = Dot(woLocal, hLocal);
		// D
		float Dval = ShadingHelper::Dggx(hLocal, alpha);
		float Jh = 1.0f / (4.f * dotWH);
		float pdfH = Dval * hLocal.z; // half-vector pdf in local
		float pdfVal = pdfH * Jh;

		outPdf = pdfVal; // 注：对折射分支，还需乘Jacobian, 这里简化

		// 6) outColor 先设置 Fresnel / (or 1-F) * albedo ...
		if (reflectBranch) {
			outColor = Colour(F,F,F); // 也可 * albedo->sample()
		}
		else {
			// 折射系数(1-F) * possibly eta^2 ...
			float ratio = (1.f - F);
			// 也可 ×(eta^2) + albedo, 视你需求
			outColor = Colour(ratio, ratio, ratio);
		}

		// 若 inside => flip back
		if (!isFront) {
			wiLocal.z = -wiLocal.z;
		}

		return shadingData.frame.toWorld(wiLocal);
	}

	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override
	{
		// 这里需要同时考虑反射项 + 折射项
		// 但往往对纯delta(或半delta)的BSDF，这里返回0或者CookTorrance
		// 简化：若都是delta => evaluate=0
		return Colour(0.f, 0.f, 0.f);
	}

	float PDF(const ShadingData& shadingData, const Vec3& wi) override
	{
		// 同 sample 中的pdfVal
		// 仅反射/折射算delta => 这里可简单返回0 or same logic
		return 0.f;
	}

	bool isPureSpecular() override { return true; }
	bool isTwoSided() override { return true; }
	float mask(const ShadingData& /*sd*/) override { return 1.0f; }
};


class OrenNayarBSDF : public BSDF
{
public:
	Texture* albedo; // 漫反射贴图
	float sigma;     // 表面粗糙度(标准差)

	OrenNayarBSDF() = default;
	OrenNayarBSDF(Texture* _albedo, float _sigma)
	{
		albedo = _albedo;
		sigma = _sigma;
	}

	// 1) 与Diffue类似，用余弦加权半球采样
	Vec3 sample(const ShadingData& shadingData, Sampler* sampler,
		Colour& reflectedColour, float& pdf) override
	{
		Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(sampler->next(),
			sampler->next());
		pdf = SamplingDistributions::cosineHemispherePDF(wiLocal);

		// 暂且先给一个基础“反射系数”
		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;

		return shadingData.frame.toWorld(wiLocal);
	}

	// 2) 用 Oren-Nayar 公式来evaluate
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override
	{
		// 将输入和输出方向都转换到局部坐标系
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);

		// 若在下半球，则无贡献
		if (woLocal.z <= 0.0f || wiLocal.z <= 0.0f)
			return Colour(0.0f, 0.0f, 0.0f);

		// 计算θi, θo, φi, φo
		float theta_i = acosf(std::max(0.0f, wiLocal.z));
		float theta_o = acosf(std::max(0.0f, woLocal.z));
		float phi_i = atan2f(wiLocal.y, wiLocal.x);
		float phi_o = atan2f(woLocal.y, woLocal.x);

		// Oren-Nayar参数 A, B
		float sigma2 = sigma * sigma;
		float A = 1.0f - (sigma2 / (2.0f * (sigma2 + 0.33f)));
		float B = 0.45f * sigma2 / (sigma2 + 0.09f);

		// 余项
		float cosDeltaPhi = cosf(phi_i - phi_o);
		float alpha = std::max(theta_i, theta_o);
		float beta = std::min(theta_i, theta_o);

		float term = std::max(0.0f, cosDeltaPhi) * sinf(alpha) * tanf(beta);

		float orenNayarValue = A + B * term;

		// 最终： ρ(x)/π * OrenNayarValue
		Colour baseColor = albedo->sample(shadingData.tu, shadingData.tv);
		return (baseColor / M_PI) * orenNayarValue;
	}

	// 3) 与cosine hemisphere采样一致
	float PDF(const ShadingData& shadingData, const Vec3& wi) override
	{
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		return SamplingDistributions::cosineHemispherePDF(wiLocal);
	}

	bool isPureSpecular() override { return false; }
	bool isTwoSided()     override { return true; }
	float mask(const ShadingData& shadingData) override
	{
		return albedo->sampleAlpha(shadingData.tu, shadingData.tv);
	}
};




class PlasticBSDF : public BSDF
{
public:
	Texture* albedo;  // 漫反射颜色贴图
	float intIOR;     // 内部折射率 (用于 Fresnel)
	float extIOR;     // 外部折射率
	float roughness;  // 用于 GGX 的粗糙度 α
	float specularWeight; // 你可以添加一个 0~1 的参数，用于指定高光在混合中的比重

	PlasticBSDF(Texture* _albedo, float _intIOR, float _extIOR,
		float _roughness, float _specW = 0.04f)
	{
		albedo = _albedo;
		intIOR = _intIOR;
		extIOR = _extIOR;
		roughness = _roughness;
		specularWeight = _specW; // 0.04 相当于常见塑料的小高光
	}

	Vec3 sample(const ShadingData& shadingData,
		Sampler* sampler,
		Colour& outColor,
		float& outPdf) override
	{
		// =========== 1) 先随机决定采样哪种分布 ==========
		float r = sampler->next();
		bool chooseSpecular = (r < specularWeight);
		// 如果 specularWeight很小, 大概率走diffuse, 否则走spec

		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		// 如果 woLocal.z<0，翻转
		if (woLocal.z < 0) woLocal.z = -woLocal.z;

		// =========== 2) 漫反射分支 =============
		if (!chooseSpecular)
		{
			// (a) 余弦加权采样
			float r1 = sampler->next();
			float r2 = sampler->next();
			Vec3 wiLocal = SamplingDistributions::cosineSampleHemisphere(r1, r2);
			// (b) PDF
			float pdfDiff = SamplingDistributions::cosineHemispherePDF(wiLocal);
			outPdf = mixPDF(pdfDiff, 0.0f);
			// 这里可直接 outPdf = pdfDiff*(1 - specularWeight) + 0 ?

			// (c) 颜色 = (albedo/π)
			outColor = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;

			// (d) 转到世界坐标
			Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
			return wiWorld;
		}
		else
		{
			// =========== 3) GGX 镜面分支 ==========
			float alphaClamp = std::max(0.001f, roughness);
			// (a) 采样半向量 hLocal
			float r1 = sampler->next();
			float r2 = sampler->next();
			// 根据GGX公式: 
			float cosThetaM = sqrtf((1.f - r1) / (r1 * (alphaClamp * alphaClamp - 1.f) + 1.f));
			float sinThetaM = sqrtf(std::max(0.f, 1.f - cosThetaM * cosThetaM));
			float phiM = 2.f * M_PI * r2;
			Vec3 hLocal(sinThetaM * cosf(phiM), sinThetaM * sinf(phiM), cosThetaM);
			// 让hLocal和 woLocal 同侧
			if (Dot(woLocal, hLocal) < 0.f) hLocal = -hLocal;

			// (b) 计算反射方向 wiLocal = reflect(-woLocal,hLocal)
			float dotWH = Dot(woLocal, hLocal);
			Vec3 wiLocal = -woLocal + 2.f * dotWH * hLocal;
			if (wiLocal.z < 0.f) {
				outPdf = 0.f;
				outColor = Colour(0.f, 0.f, 0.f);
				return shadingData.frame.toWorld(wiLocal);
			}

			// (c) PDF(half-vector)
			float Dval = ShadingHelper::Dggx(hLocal, alphaClamp);
			float pdfHalf = Dval * hLocal.z / (4.f * dotWH);
			// 整体pdf = specularWeight * pdfHalf + ...
			// 但我们只走spec branch => outPdf = pdfHalf*(some weighting)
			// For simplicity:
			outPdf = pdfHalf;
			// (d) 颜色: CookTorrance需要 Fresnel( dotWH ),这里可简化
			float cosOH = dotWH;
			float F = ShadingHelper::fresnelDielectric(cosOH, intIOR, extIOR);
			outColor = Colour(F,F,F);
			// 也可 × alpha / something ...

			Vec3 wiWorld = shadingData.frame.toWorld(wiLocal);
			return wiWorld;
		}
	}


	// Evaluate the BSDF for a given incoming ray direction
	Colour evaluate(const ShadingData& shadingData, const Vec3& wi) override
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.f || wiLocal.z <= 0.f)
			return Colour(0.f, 0.f, 0.f);

		// 1) Diffuse part = (1 - specWeight)*(albedo/π)
		Colour kd = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
		kd = kd * (1.f - specularWeight);

		// 2) GGX part
		//   half-vector
		Vec3 hLocal = (wiLocal + woLocal).normalize();
		if (hLocal.z <= 0.f)
			return kd; // no highlight
		float alphaClamp = std::max(0.001f, roughness);
		float D = ShadingHelper::Dggx(hLocal, alphaClamp);
		float G = ShadingHelper::Gggx(wiLocal, woLocal, alphaClamp);
		float dotWH = Dot(woLocal, hLocal);
		float cosOH = dotWH;
		// Fresnel
		float F = ShadingHelper::fresnelDielectric(cosOH, intIOR, extIOR);

		float denom = 4.f * wiLocal.z * woLocal.z + 1e-6f;
		Colour spec = D * G * F / denom;
		// 并乘以 specularWeight
		spec = spec * specularWeight;

		return kd + spec;
	}


	float PDF(const ShadingData& shadingData, const Vec3& wi) override
	{
		Vec3 woLocal = shadingData.frame.toLocal(shadingData.wo);
		Vec3 wiLocal = shadingData.frame.toLocal(wi);
		if (woLocal.z <= 0.f || wiLocal.z <= 0.f)
			return 0.f;

		// (1) Diffuse PDF
		float pdfDiff = SamplingDistributions::cosineHemispherePDF(wiLocal); // = wiLocal.z/π

		// (2) GGX PDF
		Vec3 hLocal = (wiLocal + woLocal).normalize();
		float dotWH = Dot(woLocal, hLocal);
		float alphaClamp = std::max(0.001f, roughness);
		float D = ShadingHelper::Dggx(hLocal, alphaClamp);
		float pdfSpec = (D * hLocal.z) / (4.f * dotWH + 1e-6f);

		// (3) Weighted sum
		float pdfMix = (1.f - specularWeight) * pdfDiff + specularWeight * pdfSpec;
		return pdfMix;
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

//class ConductorBSDF : public BSDF
//{
//public:
//	Texture* albedo;
//	Colour eta;
//	Colour k;
//	float alpha;
//	ConductorBSDF() = default;
//	ConductorBSDF(Texture* _albedo, Colour _eta, Colour _k, float roughness)
//	{
//		albedo = _albedo;
//		eta = _eta;
//		k = _k;
//		alpha = 1.62142f * sqrtf(roughness);
//	}
//	Vec3 sample(const ShadingData& shadingData, Sampler* sampler, Colour& reflectedColour, float& pdf)
//	{
//		// Replace this with Conductor sampling code
//		Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
//		pdf = wi.z / M_PI;
//		reflectedColour = albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
//		wi = shadingData.frame.toWorld(wi);
//		return wi;
//	}
//	Colour evaluate(const ShadingData& shadingData, const Vec3& wi)
//	{
//		// Replace this with Conductor evaluation code
//		return albedo->sample(shadingData.tu, shadingData.tv) / M_PI;
//	}
//	float PDF(const ShadingData& shadingData, const Vec3& wi)
//	{
//		// Replace this with Conductor PDF
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