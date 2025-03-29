#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include <thread>
#include <functional>

#define MAX_DEPTH 5

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	int numProcs;
	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		// film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new GaussianFilter(2, 1));
		//film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new MitchellNetravaliFilter());
		SYSTEM_INFO sysInfo;
		GetSystemInfo(&sysInfo);
		numProcs = sysInfo.dwNumberOfProcessors;
		threads = new std::thread*[numProcs];
		samplers = new MTRandom[numProcs];
		clear();
	}
	void clear()
	{
		film->clear();
	}
	Colour computeDirect(ShadingData shadingData, Sampler* sampler)
	{
		if (shadingData.bsdf->isPureSpecular() == true)
		{
			return Colour(0.0f, 0.0f, 0.0f);
		}
		// Sample a light
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		// Sample a point on the light
		float pdf;
		Colour emitted;
		Vec3 p = light->sample(shadingData, sampler, emitted, pdf);
		if (light->isArea())
		{
			// Calculate GTerm
			Vec3 wi = p - shadingData.x;
			float l = wi.lengthSq();
			wi = wi.normalize();
			float GTerm = (max(Dot(wi, shadingData.sNormal), 0.0f) * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / l;
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, p))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		else
		{
			// Calculate GTerm
			Vec3 wi = p;
			float GTerm = max(Dot(wi, shadingData.sNormal), 0.0f);
			if (GTerm > 0)
			{
				// Trace
				if (scene->visible(shadingData.x, shadingData.x + (p * 10000.0f)))
				{
					// Shade
					return shadingData.bsdf->evaluate(shadingData, wi) * emitted * GTerm / (pmf * pdf);
				}
			}
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	Colour pathTrace(Ray& r, Colour& pathThroughput, int depth, Sampler* sampler, bool canHitLight = true)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				if (canHitLight == true)
				{
					return pathThroughput * shadingData.bsdf->emit(shadingData, shadingData.wo);
				}
				else
				{
					return Colour(0.0f, 0.0f, 0.0f);
				}
			}
			Colour direct = pathThroughput * computeDirect(shadingData, sampler);
			if (depth > MAX_DEPTH)
			{
				return direct;
			}
			float russianRouletteProbability = min(pathThroughput.Lum(), 0.9f);
			if (sampler->next() < russianRouletteProbability)
			{
				pathThroughput = pathThroughput / russianRouletteProbability;
			}
			else
			{
				return direct;
			}
			Colour bsdf;
			float pdf;
			Vec3 wi = SamplingDistributions::cosineSampleHemisphere(sampler->next(), sampler->next());
			pdf = SamplingDistributions::cosineHemispherePDF(wi);
			wi = shadingData.frame.toWorld(wi);
			bsdf = shadingData.bsdf->evaluate(shadingData, wi);
			pathThroughput = pathThroughput * bsdf * fabsf(Dot(wi, shadingData.sNormal)) / pdf;
			r.init(shadingData.x + (wi * EPSILON), wi);
			return (direct + pathTrace(r, pathThroughput, depth + 1, sampler, shadingData.bsdf->isPureSpecular()));
		}
		return scene->background->evaluate(r.dir);
	}

	Colour pathTraceMIS(Ray& r, Colour pathThroughput, int depth, Sampler* sampler)
	{
		// 如果超过最大深度，直接返回黑色
		if (depth >= MAX_DEPTH)
			return Colour(0.0f, 0.0f, 0.0f);

		// 1. 场景求交
		IntersectionData intersection = scene->traverse(r);
		// 若没击中任何物体，则返回背景
		if (intersection.t == FLT_MAX) {
			return pathThroughput * scene->background->evaluate(r.dir);
		}

		// 2. 计算着色信息
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		BSDF* bsdf = shadingData.bsdf;

		// 3. 如若打到光源本身
		//    常见做法：若是首个 hit，或者纯镜面弹射一路过来，直接收光；否则交给 “灯光采样” 来算
		if (bsdf->isLight()) {
			// 简单处理：若 depth == 0 或上一次是镜面，也可以加上自发光
			// （按需求可自行精调）
			if (depth == 0 || bsdf->isPureSpecular()) {
				return pathThroughput * bsdf->emit(shadingData, shadingData.wo);
			}
			else {
				return Colour(0.0f, 0.0f, 0.0f);
			}
		}

		// -----------------------------------------------------------------------------------
		// Part A: 灯光采样 (Next-Event Estimation) + MIS
		// -----------------------------------------------------------------------------------
		Colour L_direct(0.0f, 0.0f, 0.0f);

		{
			// 从场景所有灯光里等概率选一个
			float lightPMF;
			Light* light = scene->sampleLight(sampler, lightPMF);
			// scene->sampleLight() 里你传回 pmf = 1.0f / lights.size()。

			// 在该灯光上采样一点
			float lightPdf;
			Colour Lemit;
			Vec3 lightSamplePos = light->sample(shadingData, sampler, Lemit, lightPdf);

			if (lightPdf > 1e-6f && lightPMF > 1e-6f && Lemit.Lum() > 0.0f)
			{
				// wi = 命中点 -> 光源点
				Vec3 wi = lightSamplePos - shadingData.x;
				float dist2 = wi.lengthSq();
				if (dist2 > 1e-10f) {
					float dist = sqrtf(dist2);
					wi /= dist; // 归一化

					float NoL = Dot(shadingData.sNormal, wi);
					if (NoL > 0.0f) {
						// 判断可见性
						if (scene->visible(shadingData.x, lightSamplePos)) {
							// BSDF 的 f(wo, wi)
							Colour f = bsdf->evaluate(shadingData, wi);
							// BSDF 的 pdf(wo->wi)，下边 MIS 要用
							float bsdfPdf = bsdf->PDF(shadingData, wi);

							// 若是面光源，还要乘上光源面的余弦项
							float LN = 1.0f;
							if (light->isArea()) {
								Vec3 nLight = light->normal(shadingData, wi);
								LN = max(-Dot(wi, nLight), 0.0f);
							}

							// 几何项
							float G = (NoL * LN) / dist2;

							// 光源采样得到的整条方向 pdf = lightPdf * lightPMF
							float pdfLight = lightPdf * lightPMF;
							// 计算 MIS 权重
							float misW = balanceHeuristic(pdfLight, bsdfPdf);

							// 累加本次直接光
							L_direct = f * Lemit * G * misW / pdfLight;
						}
					}
				}
			}
		}

		// -----------------------------------------------------------------------------------
		// Part B: 俄罗斯轮盘
		// -----------------------------------------------------------------------------------
		float rrProb = min(pathThroughput.Lum(), 0.9f);
		if (sampler->next() > rrProb) {
			// 不再弹射，返回目前的直接光
			return pathThroughput * L_direct;
		}
		// 否则继续，但要除存活概率
		pathThroughput /= rrProb;

		// -----------------------------------------------------------------------------------
		// Part C: BSDF 采样
		// -----------------------------------------------------------------------------------
		Colour bsdfVal;
		float bsdfPdf;
		// sample(...) 里会返回新的方向 wiBSDF，并把 f(wo->wiBSDF)/pdf 之类放到 bsdfVal 也行；
		// 但目前你的代码是 “reflectedColour” = bsdfVal
		Vec3 wiBSDF = bsdf->sample(shadingData, sampler, bsdfVal, bsdfPdf);

		if (bsdfPdf < 1e-6f) {
			// PDF太小，直接返回
			return pathThroughput * L_direct;
		}

		// 看采样到的方向相对于表面法线
		float cosTerm = Dot(shadingData.sNormal, wiBSDF);
		if (cosTerm < 0.0f && !bsdf->isTwoSided()) {
			// 单面物体 + 采样到背面 => 无贡献
			return pathThroughput * L_direct;
		}
		cosTerm = fabsf(cosTerm);

		// -----------------------------------------------------------------------------------
		// 看这一跳是否直接打到光源 => 加灯的辐射 (也要做 MIS)
		// -----------------------------------------------------------------------------------
		Colour L_bsdfLight(0.0f, 0.0f, 0.0f);

		{
			Ray shadowRay(shadingData.x + wiBSDF * EPSILON, wiBSDF);
			IntersectionData lightHit = scene->traverse(shadowRay);
			if (lightHit.t < FLT_MAX) {
				ShadingData lightSD = scene->calculateShadingData(lightHit, shadowRay);
				if (lightSD.bsdf->isLight()) {
					// 取出光源的自发光
					Colour Le = lightSD.bsdf->emit(lightSD, lightSD.wo);

					// 计算“灯光采样”那边的 pdf => lightPdf2 = <灯本身的 pdf> * <选中此灯的 pmf>
					Light* hitLight = dynamic_cast<Light*>(lightSD.bsdf);
					if (hitLight) {
						// 例如：
						float lightPdf2 = hitLight->PDF(lightSD, -wiBSDF)
							* (1.0f / (float)scene->lights.size());
						float misW = balanceHeuristic(bsdfPdf, lightPdf2);
						L_bsdfLight = Le * misW;
					}
				}
			}
		}

		// -----------------------------------------------------------------------------------
		// 更新 pathThroughput：因为我们又经历了一次 BSDF 乘 cos / pdf
		// -----------------------------------------------------------------------------------
		pathThroughput *= (bsdfVal * (cosTerm / bsdfPdf));

		// -----------------------------------------------------------------------------------
		// 递归追下一弹
		r.init(shadingData.x + wiBSDF * EPSILON, wiBSDF);
		Colour L_next = pathTraceMIS(r, pathThroughput, depth + 1, sampler);

		// -----------------------------------------------------------------------------------
		// 返回：直接光 + (本次若击中灯光的那点发射) + 后续弹射
		// 注意灯光采样那部分乘“旧的 pathThroughput”，
		// 而 BSDF 命中光源 + 后面递归都乘“更新后的” throughput
		// 这里把加和写在一起看清楚
		//  => L_direct 用旧的 pathThroughput
		//  => L_bsdfLight 和 L_next 都用新的 pathThroughput
		return (pathThroughput * L_direct) + (pathThroughput * L_bsdfLight) + L_next;
	}



	Colour direct(Ray& r, Sampler* sampler)
	{
		// Compute direct lighting for an image sampler here
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return computeDirect(shadingData, sampler);
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	Colour albedo(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		if (shadingData.t < FLT_MAX)
		{
			if (shadingData.bsdf->isLight())
			{
				return shadingData.bsdf->emit(shadingData, shadingData.wo);
			}
			return shadingData.bsdf->evaluate(shadingData, Vec3(0, 1, 0));
		}
		return scene->background->evaluate(r.dir);
	}
	Colour viewNormals(Ray& r)
	{
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t < FLT_MAX)
		{
			ShadingData shadingData = scene->calculateShadingData(intersection, r);
			return Colour(fabsf(shadingData.sNormal.x), fabsf(shadingData.sNormal.y), fabsf(shadingData.sNormal.z));
		}
		return Colour(0.0f, 0.0f, 0.0f);
	}

	float computeTileVariance(Film* film, int startX, int startY, int endX, int endY)
	{
		Colour sum(0.0f, 0.0f, 0.0f), sumSq(0.0f, 0.0f, 0.0f);
		int count = 0;
		for (int y = startY; y < endY; y++)
		{
			for (int x = startX; x < endX; x++)
			{
				int idx = y * film->width + x;
				Colour c = film->film[idx] / (float)film->SPP; // 求平均颜色
				sum = sum + c;
				sumSq = sumSq + (c * c);
				count++;
			}
		}
		if (count == 0)
			return 0.0f;
		Colour mean = sum / (float)count;
		Colour meanSq = sumSq / (float)count;
		float variance = ((meanSq.r - mean.r * mean.r) +
			(meanSq.g - mean.g * mean.g) +
			(meanSq.b - mean.b * mean.b)) / 3.0f;
		return variance;
	}

	inline float balanceHeuristic(float pdfA, float pdfB)
	{
		float denom = pdfA + pdfB;
		// 防止极小数除零
		if (denom < 1e-8f) return 0.0f;
		return pdfA / denom;
	}

	void render()
	{
		film->incrementSPP();

		const int tileSize = 16;  // 细粒度 tile
		const int tileCountX = (film->width + tileSize - 1) / tileSize;
		const int tileCountY = (film->height + tileSize - 1) / tileSize;
		const int totalTiles = tileCountX * tileCountY;

		// 为每个 tile 设置收敛标记（初始均未收敛）
		std::vector<std::atomic<bool>> tileFinished(totalTiles);
		for (int i = 0; i < totalTiles; i++)
			tileFinished[i].store(false);

		// 动态任务调度的原子计数器
		std::atomic<int> nextTileIndex(0);
		// 设定收敛阈值（根据场景与采样策略调参）
		const float varianceThreshold = 0.001f;

		// lambda：渲染单个 tile，同时在渲染完后检查方差
		auto renderTileAdaptive = [&](int tileIndex, Sampler* threadSampler)
			{
				int tileX = tileIndex % tileCountX;
				int tileY = tileIndex / tileCountX;
				int startX = tileX * tileSize;
				int startY = tileY * tileSize;
				int endX = min(startX + tileSize, (int)film->width);
				int endY = min(startY + tileSize, (int)film->height);

				// 如果该 tile 已经收敛，则跳过
				if (tileFinished[tileIndex].load())
					return;

				// 遍历该 tile 内所有像素进行采样
				for (int y = startY; y < endY; y++)
				{
					for (int x = startX; x < endX; x++)
					{
						float px = x + 0.5f;
						float py = y + 0.5f;
						Ray ray = scene->camera.generateRay(px, py);
						Colour throughput(1.0f, 1.0f, 1.0f);
						// 这里调用 pathTrace()；同时可以扩展，将 albedo 和 normal 信息一并采样
						//Colour col = pathTrace(ray, throughput, 0, threadSampler);
						Colour col = pathTraceMIS(ray, throughput, 0, threadSampler);
						// 假设这里我们调用 splat() 只写入颜色，
						// 真实场景中你应同时传入对应的 albedo 与法线数据（此处示例简单）
						Colour normalColour(scene->camera.origin.x, scene->camera.origin.y, scene->camera.origin.z);
						film->splat(px, py, col);
						// 注：这里用 col 作为 albedo（仅示例），用 camera.origin 代替法线（仅占位）
						unsigned char r = (unsigned char)(col.r * 255);
						unsigned char g = (unsigned char)(col.g * 255);
						unsigned char b = (unsigned char)(col.b * 255);
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
					}
				}
				// 渲染完 tile 后计算方差
				float var = computeTileVariance(film, startX, startY, endX, endY);
				if (var < varianceThreshold)
					tileFinished[tileIndex].store(true);
			};

		// 创建线程池，动态领取 tile 任务
		std::vector<std::thread*> threadPool(numProcs);
		for (int i = 0; i < numProcs; i++)
		{
			threadPool[i] = new std::thread([=, &nextTileIndex, &tileFinished]()
				{
					Sampler* threadSampler = &samplers[i];
					while (true)
					{
						int tileIndex = nextTileIndex.fetch_add(1);
						if (tileIndex >= totalTiles)
							break;
						renderTileAdaptive(tileIndex, threadSampler);
					}
				});
		}
		for (int i = 0; i < numProcs; i++)
		{
			threadPool[i]->join();
			delete threadPool[i];
		}
	}
	int getSPP()
	{
		return film->SPP;
	}
	void saveHDR(std::string filename)
	{
		film->save(filename);
	}
	void savePNG(std::string filename)
	{
		stbi_write_png(filename.c_str(), canvas->getWidth(), canvas->getHeight(), 3, canvas->getBackBuffer(), canvas->getWidth() * 3);
	}
};