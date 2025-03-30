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
	int tileSize = 16;  // Size of each tile
	int tileCountX;     // Number of tiles in the x direction
	int tileCountY;     // Number of tiles in the y direction
	std::vector<std::atomic<bool>> tileFinished; // For tracking if a tile has converged
	int minSamples = 1;  // Minimum samples per tile
	int maxSamples = 16; // Maximum samples per tile
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
		// 场景交点求交
		IntersectionData intersection = scene->traverse(r);
		if (intersection.t == FLT_MAX)
			return pathThroughput * scene->background->evaluate(r.dir);

		ShadingData shadingData = scene->calculateShadingData(intersection, r);
		BSDF* bsdf = shadingData.bsdf;
		int N_lightSamples = 4;
		// 直接光照计算：光源采样 + MIS
		Colour L_direct(0.0f, 0.0f, 0.0f);
		{
			// 从所有光源中等概率选择一个
			float lightPMF;
			Light* light = scene->sampleLight(sampler, lightPMF);
			float lightPdf;
			Colour Lemit;
			Vec3 lightSamplePos = light->sample(shadingData, sampler, Lemit, lightPdf);

			if (lightPdf > 1e-6f && lightPMF > 1e-6f && Lemit.Lum() > 0.0f)
			{
				// 计算从物体到光源的方向 wi
				Vec3 wi = lightSamplePos - shadingData.x;
				float dist2 = wi.lengthSq();
				if (dist2 > 1e-10f) {
					float dist = sqrtf(dist2);
					wi /= dist;

					// 判断是否能从物体表面看到光源（可见性）
					if (scene->visible(shadingData.x, lightSamplePos))
					{
						// BSDF 评估和PDF
						Colour f = bsdf->evaluate(shadingData, wi);
						float bsdfPdf = bsdf->PDF(shadingData, wi);

						// 计算几何项
						float NoL = Dot(shadingData.sNormal, wi);
						if (NoL > 0.0f) {
							float G = (NoL * max(-Dot(wi, light->normal(shadingData, wi)), 0.0f)) / dist2;

							// 计算MIS权重
							float pdfLight = lightPdf * lightPMF;
							float misW = balanceHeuristic(pdfLight, bsdfPdf);

							// 累加直接光照
							L_direct += (f * Lemit * G) * (misW / pdfLight);
						}
					}
				}
				L_direct /= float(N_lightSamples);
			}
		}

		// 俄罗斯轮盘：决定是否继续反射
		float rrProb = min(pathThroughput.Lum(), 0.9f);
		if (sampler->next() > rrProb) {
			return pathThroughput * L_direct;
		}
		pathThroughput /= rrProb;

		// 反射BSDF采样
		Colour bsdfVal;
		float bsdfPdf;
		Vec3 wiBSDF = bsdf->sample(shadingData, sampler, bsdfVal, bsdfPdf);

		if (bsdfPdf < 1e-6f) {
			return pathThroughput * L_direct;
		}

		// 对于采样方向，计算光源采样加权
		float cosTerm = Dot(shadingData.sNormal, wiBSDF);
		if (cosTerm < 0.0f && !bsdf->isTwoSided()) {
			return pathThroughput * L_direct;
		}
		cosTerm = fabsf(cosTerm);

		Colour L_bsdfLight(0.0f, 0.0f, 0.0f);
		// 计算采样到光源的情况（例如，来自BSDF的反射光线直接命中光源）
		Ray shadowRay(shadingData.x + wiBSDF * EPSILON, wiBSDF);
		IntersectionData lightHit = scene->traverse(shadowRay);
		if (lightHit.t < FLT_MAX) {
			ShadingData lightSD = scene->calculateShadingData(lightHit, shadowRay);
			if (lightSD.bsdf->isLight()) {
				Colour Le = lightSD.bsdf->emit(lightSD, lightSD.wo);
				Light* hitLight = dynamic_cast<Light*>(lightSD.bsdf);
				if (hitLight) {
					float lightPdf2 = hitLight->PDF(lightSD, -wiBSDF) * (1.0f / (float)scene->lights.size());
					float misW = balanceHeuristic(bsdfPdf, lightPdf2);
					L_bsdfLight = Le * misW;
				}
			}
		}

		// 更新路径通量
		pathThroughput *= (bsdfVal * (cosTerm / bsdfPdf));

		// 递归计算下一跳路径
		r.init(shadingData.x + wiBSDF * EPSILON, wiBSDF);
		Colour L_next = pathTraceMIS(r, pathThroughput, depth + 1, sampler);

		// 返回最终的光照（直接光、BSDF光源光照、递归光照）
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

	inline float balanceHeuristic(float pdfA, float pdfB)
	{
		float denom = pdfA + pdfB;
		// 防止极小数除零
		if (denom < 1e-8f) return 0.0f;
		return pdfA / denom;
	}
	// 更新 computeTileVariance 来计算块内方差并进行平滑处理
	float computeBlockVariance(Film* film, int startX, int startY, int endX, int endY)
	{
		Colour sum(0.0f, 0.0f, 0.0f), sumSq(0.0f, 0.0f, 0.0f);
		int count = 0;

		// 确保开始和结束索引在有效范围内
		startX = max(0, startX);
		startY = max(0, startY);
		endX = min(film->width, endX);
		endY = min(film->height, endY);

		for (int y = startY; y < endY; y++)
		{
			for (int x = startX; x < endX; x++)
			{
				int idx = y * film->width + x;
				if (idx < 0 || idx >= film->width * film->height) {
					continue;  // Skip invalid index
				}
				Colour c = film->film[idx] / (float)film->SPP; // 平均颜色
				sum = sum + c;
				sumSq = sumSq + (c * c);
				count++;
			}
		}

		if (count == 0)
			return 0.0f;

		// 计算平均值和平方平均值
		Colour mean = sum / (float)count;
		Colour meanSq = sumSq / (float)count;

		// 计算方差并返回
		float variance = ((meanSq.r - mean.r * mean.r) +
			(meanSq.g - mean.g * mean.g) +
			(meanSq.b - mean.b * mean.b)) / 3.0f;
		return variance;
	}


	// 使用平滑处理来平滑方差值
	float smoothVariance(Film* film, int startX, int startY, int endX, int endY)
	{
		float totalVariance = 0.0f;
		int numBlocks = 0;
		int smoothRadius = 2; // 平滑半径，控制邻域范围

		// 保证 x, y 在有效范围内
		startX = max(0, startX - smoothRadius);
		startY = max(0, startY - smoothRadius);
		endX = min(film->width, endX + smoothRadius);
		endY = min(film->height, endY + smoothRadius);

		for (int y = startY; y <= endY; y++)
		{
			for (int x = startX; x <= endX; x++)
			{
				if (x >= 0 && x < film->width && y >= 0 && y < film->height)
				{
					totalVariance += computeBlockVariance(film, x, y, x + smoothRadius, y + smoothRadius);
					numBlocks++;
				}
			}
		}

		return numBlocks > 0 ? totalVariance / numBlocks : 0.0f;
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
		const float varianceThreshold = 0.001f;  // 可调整阈值

		// lambda：渲染单个 tile，同时在渲染完后检查方差
		auto renderTileAdaptive = [&](int tileIndex, Sampler* threadSampler)
			{
				int tileX = tileIndex % tileCountX;
				int tileY = tileIndex / tileCountX;
				int startX = tileX * tileSize;
				int startY = tileY * tileSize;
				int endX = min(startX + tileSize, (int)film->width);
				int endY = min(startY + tileSize, (int)film->height);

				// 处理像素的采样，注意检查边界
				for (int y = startY; y < endY; y++)
				{
					for (int x = startX; x < endX; x++)
					{
						float px = x + 0.5f;
						float py = y + 0.5f;
						Ray ray = scene->camera.generateRay(px, py);
						Colour throughput(1.0f, 1.0f, 1.0f);
						Colour col = pathTraceMIS(ray, throughput, 0, threadSampler);
						film->splat(px, py, col); // 将采样结果写入 film
						unsigned char r = (unsigned char)(col.r * 255);
						unsigned char g = (unsigned char)(col.g * 255);
						unsigned char b = (unsigned char)(col.b * 255);
						film->tonemap(x, y, r, g, b);
						canvas->draw(x, y, r, g, b);
					}
				}

				// 计算方差并应用平滑
				float var = smoothVariance(film, startX, startY, endX, endY);
				if (var < varianceThreshold)
				{
					tileFinished[tileIndex].store(true);  // 如果方差小于阈值，标记 tile 完成
				}
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

		// 等待所有线程完成
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