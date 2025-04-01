#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "GamesEngineeringBase.h"
#include "VPL.h"
#include <thread>
#include <functional>
#include <vector>
#include <mutex>
#include <atomic>
#include <iostream> 
#define MAX_DEPTH 5
#define VPL_EPSILON 1e-4f 

class RayTracer
{
public:
	Scene* scene;
	GamesEngineeringBase::Window* canvas;
	Film* film;
	MTRandom *samplers;
	std::thread **threads;
	std::mutex vplMutex;
	int numProcs;
	void init(Scene* _scene, GamesEngineeringBase::Window* _canvas)
	{
		scene = _scene;
		canvas = _canvas;
		film = new Film();
		// film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
		film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new GaussianFilter(2, 1));
		//film->init((unsigned int)scene->camera.width, (unsigned int)scene->camera.height, new BoxFilter());
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
	Colour computeDirect(const ShadingData& sData, Sampler* sampler) {
		// Pure delta
		if (sData.bsdf->isPureSpecular())
			return Colour(0.f, 0.f, 0.f);

		// Random sample
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		if (!light)
			return Colour(0.f, 0.f, 0.f);

		float lightPdf;
		Colour Lemit;
		Vec3 sampleVal = light->sample(sData, sampler, Lemit, lightPdf);
		float pdfLight = lightPdf * pmf;
		if (pdfLight < 1e-8f || Lemit.Lum() < 1e-8f)
			return Colour(0.f, 0.f, 0.f);

		// Area light / Env map
		Vec3 wi;
		float G = 0.f;
		bool visible = false;
		if (light->isArea()) {
			// Area light：sampleVal p
			wi = sampleVal - sData.x;
			float dist2 = wi.lengthSq();
			if (dist2 > 1e-10f) {
				float dist = sqrtf(dist2);
				wi /= dist;
				float NoL = Dot(sData.sNormal, wi);
				if (NoL > 0.f) {
					float cosLight = max(-Dot(wi, light->normal(sData, wi)), 0.f);
					G = (NoL * cosLight) / dist2;
					visible = scene->visible(sData.x, sampleVal);
				}
			}
		}
		else {
			// Env map sampleVal dir
			wi = sampleVal;
			float NoL = Dot(sData.sNormal, wi);
			if (NoL > 0.f) {
				G = NoL;
				visible = scene->visible(sData.x, sData.x + wi * 1e6f);
			}
		}
		if (!visible || G < 1e-12f)
			return Colour(0.f, 0.f, 0.f);

		// BSDF and PDF
		Colour f = sData.bsdf->evaluate(sData, wi);
		float bsdfPdf = sData.bsdf->PDF(sData, wi);
		if (bsdfPdf < 1e-8f || f.Lum() < 1e-8f)
			return Colour(0.f, 0.f, 0.f);

		float misW = balanceHeuristic(pdfLight, bsdfPdf);
		return f * Lemit * G * (misW / pdfLight);
	}

	// Trace VPLs Parallel
	void traceVPLs(int N_VPLs) {
		scene->vplList.clear();
		std::cout << "Starting VPL generation (" << N_VPLs << " initial paths across " << numProcs << " threads)..." << std::endl;
		auto start_time = std::chrono::high_resolution_clock::now();

		std::vector<std::vector<VPL>> threadVPLs(numProcs); // Local VPL lists for each thread
		std::vector<std::thread> workers(numProcs);
		std::atomic<int> initialPathsProcessed(0);

		int pathsPerThread = (N_VPLs + numProcs - 1) / numProcs; // Distribute initial paths

		for (int i = 0; i < numProcs; ++i) {
			workers[i] = std::thread([&, i, pathsPerThread]() {
				Sampler* sampler = &samplers[i];
				int startPath = i * pathsPerThread;
				int endPath = min(startPath + pathsPerThread, N_VPLs);

				for (int pathIdx = startPath; pathIdx < endPath; ++pathIdx) {
					// Sample a light source
					float lightPmf;
					Light* light = scene->sampleLight(sampler, lightPmf);
					// Ensure a valid light was sampled and pmf available
					if (!light || lightPmf < EPSILON) continue;

					// Sample position and direction from the light
					float pdfPos, pdfDir;
					Vec3 pLight = light->samplePositionFromLight(sampler, pdfPos);
					Vec3 dirLight = light->sampleDirectionFromLight(sampler, pdfDir); // Direction AWAY from light pos

					// Ensure PDFs are valid
					if (pdfPos < EPSILON || pdfDir < EPSILON) continue;

					// Get light emission Radiance and calculate initial Flux
					Colour Le = light->evaluate(-dirLight); // Radiance towards the sampled direction
					if (!Le.isValid() || Le.isBlack()) continue;

					// Calculate total probability density for initial sample
					float pdfTotalInitial = pdfPos * pdfDir * lightPmf;
					if (pdfTotalInitial < EPSILON) continue;

					// Using Radiance/PDF
					Colour currentFlux = Le / pdfTotalInitial;

					if (!currentFlux.isValid() || currentFlux.isBlack()) continue; // Check initial flux validity

					// Trace light path
					Ray currentRay(pLight + dirLight * EPSILON, dirLight);
					Colour pathThroughput = Colour(1.0f, 1.0f, 1.0f);

					for (int depth = 0; depth < MAX_DEPTH; ++depth) {
						IntersectionData isect = scene->traverse(currentRay);
						if (isect.t == FLT_MAX) break;

						ShadingData hitData = scene->calculateShadingData(isect, currentRay);
						if (!hitData.bsdf) break;

						// Store VPL if the surface is not purely specular
						if (!hitData.bsdf->isPureSpecular()) {
							Colour vplFluxToStore = currentFlux * pathThroughput;

							if (vplFluxToStore.isValid() && !vplFluxToStore.isBlack()) {
								// Store the VPL with its associated flux and shading data
								threadVPLs[i].emplace_back(hitData, vplFluxToStore);
							}
							else {
							}
						}
						// Continue the path: Sample BSDF for next direction
						Colour f_bsdf;
						float pdf_bsdf;
						Vec3 wi = hitData.bsdf->sample(hitData, sampler, f_bsdf, pdf_bsdf);

						// Check if BSDF sampling was successful
						if (pdf_bsdf < EPSILON || !f_bsdf.isValid() || f_bsdf.isBlack()) {
							break;
						}
						// Calculate cosine term > 0
						float cosTerm = fabsf(Dot(hitData.sNormal, wi));
						if (cosTerm < EPSILON) {
							break;
						}
						pathThroughput = pathThroughput * f_bsdf * cosTerm / pdf_bsdf;

						// Check throughput validity after update
						if (!pathThroughput.isValid() || pathThroughput.isBlack()) {
							break;
						}
						// Russian Roulette
						float rrProb = min(pathThroughput.Lum(), 0.95f); // Use luminance for probability
						if (!pathThroughput.isValid()) rrProb = 0.0f;
						if (sampler->next() > rrProb) {
							break; // Terminate path
						}
						if (rrProb < EPSILON) break;
						pathThroughput = pathThroughput / rrProb;

						// Check throughput validity after RR
						if (!pathThroughput.isValid() || pathThroughput.isBlack()) {
							break;
						}
						// Update ray for next bounce
						currentRay.init(hitData.x + wi * EPSILON, wi);

					}
				}
				});
		}

		int totalVPLsGenerated = 0;
		for (int i = 0; i < numProcs; ++i) {
			workers[i].join();
			totalVPLsGenerated += threadVPLs[i].size();
			scene->vplList.insert(scene->vplList.end(), threadVPLs[i].begin(), threadVPLs[i].end());
		}
		if (scene->vplList.empty() && N_VPLs > 0) {
		}

		auto end_time = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double, std::milli> elapsed_ms = end_time - start_time;
		if (scene->vplList.empty() && N_VPLs > 0) {
			std::cerr << "Warning: No VPLs were generated. Check light sources, sampling, or path tracing logic." << std::endl;
		}
		std::cout << "VPL Generation Time: " << elapsed_ms.count() << " ms" << std::endl;

		if (scene->vplList.empty()) {
			std::cerr << "Error: No VPLs generated. Rendering will likely be black or direct only." << std::endl;
		}
	}

	// Compute indirect illumination using VPLs
	Colour computeIndirectVPL(const ShadingData& sData, int N_VPLs) {
		if (N_VPLs == 0 || scene->vplList.empty() || sData.bsdf == nullptr || sData.bsdf->isPureSpecular()) {
			return Colour(0.0f, 0.0f, 0.0f);
		}

		const Vec3& p = sData.x;        // Point being shaded
		const Vec3& n = sData.sNormal;  // Normal at point being shaded
		const BSDF* bsdf = sData.bsdf;  // BSDF at point being shaded
		Colour indirectLight(0.0f, 0.0f, 0.0f);

		for (const VPL& vpl : scene->vplList) {
			const Vec3& p_vpl = vpl.shadingData.x;
			const Vec3& n_vpl = vpl.shadingData.sNormal; // Normal at VPL
			const Colour& flux_vpl = vpl.flux;         // Flux stored in the VPL

			if (!flux_vpl.isValid() || flux_vpl.isBlack()) {
				continue;
			}

			// Vector from shading point p to VPL position p_vpl
			Vec3 dir_p_to_vpl = p_vpl - p;
			float distSq = dir_p_to_vpl.lengthSq();

			// Avoid self-contribution and division by zero
			if (distSq < EPSILON * EPSILON) {
				continue;
			}

			float dist = sqrt(distSq);
			dir_p_to_vpl /= dist; // Normalized direction from p towards vpl

			if (!scene->visible(p, p_vpl)) {
				continue; // VPL is occluded
			}

			// Geometry Term
			Vec3 wi_incoming = -dir_p_to_vpl; // Normalized direction from vpl towards p

			float cosTheta_p = fabsf(Dot(n, wi_incoming));       // Cosine at shading point p (angle with incoming light)
			float cosTheta_vpl = fabsf(Dot(n_vpl, dir_p_to_vpl)); // Cosine at VPL point (angle with outgoing light towards p)

			if (cosTheta_p < EPSILON || cosTheta_vpl < EPSILON) {
				continue;
			}

			float G = cosTheta_p * cosTheta_vpl / distSq;
			if (!std::isfinite(G) || G < 0) continue;

			// BSDF Evaluation
			Colour f = bsdf->evaluate(sData, wi_incoming); // incoming dir

			if (!f.isValid() || f.isBlack()) {
				continue;
			}
			// Accumulate Contribution
			indirectLight += flux_vpl * f * G;
		}
		// Check if valid before normalize
		if (!indirectLight.isValid()) {
			return Colour(0.0f, 0.0f, 0.0f);
		}
		if (N_VPLs > 0) {
			indirectLight /= (float)N_VPLs;
		}
		else if (!scene->vplList.empty()) {
			// Fallback N_VPLs > 0
			indirectLight /= (float)scene->vplList.size();
		}


		return indirectLight;
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
	
	Colour pathTraceMIS(Ray& r, Colour throughput, int depth, Sampler* sampler, bool canHitLight = true) {
		IntersectionData isect = scene->traverse(r);
		if (isect.t == FLT_MAX)
			return throughput * scene->background->evaluate(r.dir);

		ShadingData sData = scene->calculateShadingData(isect, r);
		BSDF* bsdf = sData.bsdf;

		// If hitting a light surface
		if (bsdf->isLight()) {
			if (canHitLight)
				return throughput * bsdf->emit(sData, sData.wo);
			else
				return Colour(0.f, 0.f, 0.f);
		}

		// Light source sampling
		Colour L_direct = Colour(0.f, 0.f, 0.f);
		if (!bsdf->isPureSpecular())
			L_direct = computeDirect(sData, sampler);
		if (depth >= MAX_DEPTH) {
			return throughput * L_direct;
		}

		// RR
		float rrProb = min(throughput.Lum(), 0.95f);
		if (sampler->next() > rrProb)
			return throughput * L_direct;
		throughput /= rrProb;

		// BSDF sampling branch
		Colour fBSDF;
		float pdfBSDF;
		Vec3 wiBSDF = bsdf->sample(sData, sampler, fBSDF, pdfBSDF);

		if (pdfBSDF < 1e-8f)
			return throughput * L_direct;

		// cosTerm
		float cosTerm = fabsf(Dot(sData.sNormal, wiBSDF));

		// Updates the BSDF contribution term, calculates the final contribution and updates the throughput
		Colour bsdfContribution = fBSDF * (cosTerm / pdfBSDF);
		throughput *= bsdfContribution;

		// Secondary MIS, if the BSDF sampling ray hits the light, do a second PDF sample
		Colour L_bsdfLight = Colour(0.f, 0.f, 0.f);
		Ray shadowRay(sData.x + wiBSDF * EPSILON, wiBSDF);
		IntersectionData li = scene->traverse(shadowRay);
		if (li.t < FLT_MAX) {
			ShadingData lSD = scene->calculateShadingData(li, shadowRay);
			if (lSD.bsdf->isLight()) {
				// For area light，light by tri ID
				Light* hitLight = scene->findLightByTriangleID(li.ID);
				if (hitLight) {
					float pmf_hit = hitLight->totalIntegratedPower() / scene->totalLightPower;
					float lightPdf2 = hitLight->PDF(lSD, -wiBSDF) * pmf_hit;
					float misW = balanceHeuristic(pdfBSDF, lightPdf2);
					Colour Le = lSD.bsdf->emit(lSD, lSD.wo);
					L_bsdfLight = Le * misW;
				}
			}
		}
		else {
			// If BSDF sampling ray does not hit the geometry, it is considered to hit the environment
			float envPdf = scene->background->PDF(sData, wiBSDF);
			float misW = balanceHeuristic(pdfBSDF, envPdf);
			Colour envRadiance = scene->background->evaluate(wiBSDF);
			L_bsdfLight = envRadiance * misW;
		}
		Ray nextRay(sData.x + wiBSDF * EPSILON, wiBSDF);
		Colour L_next = pathTraceMIS(nextRay, throughput, depth + 1, sampler, false);
		return throughput * L_direct + throughput * L_bsdfLight + L_next;
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

	void connectToCamera(const Vec3& p, const Vec3& normal, const Colour& L)
	{
		// Try to project the point p to the camera's screen coordinates
		float screenX, screenY;
		bool visibleOnScreen = scene->camera.projectOntoCamera(p, screenX, screenY);
		if (!visibleOnScreen) {
			// Outside
			return;
		}

		// Calculate angle between the point and the line of sight of the camera
		Vec3 dirToCam = scene->camera.origin - p;
		float dist2 = dirToCam.lengthSq();
		if (dist2 < 1e-12f) return;
		dirToCam = dirToCam.normalize();

		float cosTheta = Dot(dirToCam, scene->camera.viewDirection);
		if (cosTheta <= 1e-5f) return;

		// Point contribution to camera
		float cosTheta2 = cosTheta * cosTheta;
		float cosTheta4 = cosTheta2 * cosTheta2;
		float We = 1.0f / (scene->camera.Afilm * max(cosTheta4, 1e-8f));

		// Write the luminous flux contribution from this point to Film
		Colour finalColor = L * We;
		film->splat(screenX, screenY, finalColor);
	}

	void lightTrace(Sampler* sampler) {
		if (scene->lights.empty()) return;

		// Random light source
		float pmf;
		Light* light = scene->sampleLight(sampler, pmf);
		if (!light) return;

		// Light surface p
		float pdfPos;
		Vec3 lightPos = light->samplePositionFromLight(sampler, pdfPos);
		pdfPos = max(pdfPos, 1e-8f);

		// Sample a dir from p pos
		float pdfDir;
		Vec3 wi = light->sampleDirectionFromLight(sampler, pdfDir);
		pdfDir = max(pdfDir, 1e-8f);

		if (pdfPos < 1e-8f || pdfDir < 1e-8f) return;

		// Le
		Colour Le = light->evaluate(-wi);

		// Calculate the PDF of the light source and combine MIS
		float combinedPdf = pdfPos * pdfDir * pmf;
		combinedPdf = max(combinedPdf, 1e-8f);

		Colour pathThroughput = Le / combinedPdf;

		// Connect to camera
		connectToCamera(lightPos, light->normal(ShadingData{}, -wi), pathThroughput);

		// Constructing a ray from the light source to the scene
		Ray r(lightPos + wi * EPSILON, wi);

		lightTracePath(r, pathThroughput, Le, sampler, 0);
	}




	void lightTracePath(Ray& r, Colour pathThroughput, Colour Le, Sampler* sampler, int depth) {
		if (depth >= MAX_DEPTH) {
			return;
		}

		// Traverse
		IntersectionData isect = scene->traverse(r);
		if (isect.t == FLT_MAX) {
			return;
		}

		// Calculate colouring data for this intersection
		ShadingData shadingData = scene->calculateShadingData(isect, r);
		BSDF* bsdf = shadingData.bsdf;

		// Connect to camera and contribute
		connectToCamera(shadingData.x, shadingData.sNormal, pathThroughput);

		if (bsdf->isLight()) {
			return;
		}

		// RR
		float rrProb = min(pathThroughput.Lum(), 0.95f);
		if (sampler->next() > rrProb) {
			return;
		}
		pathThroughput /= rrProb;

		// Sampling the direction of reflection/refraction from BSDF
		Colour fBSDF;
		float pdfBSDF;
		Vec3 wi = bsdf->sample(shadingData, sampler, fBSDF, pdfBSDF);

		if (pdfBSDF < 1e-8f) {
			return;
		}

		// Gem
		float cosTerm = std::abs(Dot(shadingData.sNormal, wi));
		Colour newThroughput = pathThroughput * fBSDF * (cosTerm / pdfBSDF);

		Ray nextRay(shadingData.x + wi * EPSILON, wi);
		lightTracePath(nextRay, newThroughput, Le, sampler, depth + 1);
	}



	inline float balanceHeuristic(float pdfA, float pdfB)
	{
		float denom = pdfA + pdfB;
		if (denom < 1e-8f) return 0.0f;
		return pdfA / denom;
	}

	// Estimate variance
	float computeBlockVariance(Film* film, int startX, int startY, int endX, int endY)
	{
		Colour sum(0.0f, 0.0f, 0.0f), sumSq(0.0f, 0.0f, 0.0f);
		int count = 0;

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
				Colour c = film->film[idx] / (float)film->SPP; // Average
				sum = sum + c;
				sumSq = sumSq + (c * c);
				count++;
			}
		}

		if (count == 0)
			return 0.0f;

		// Calculate mean and squared mean
		Colour mean = sum / (float)count;
		Colour meanSq = sumSq / (float)count;

		// Calculate the variance and return
		float variance = ((meanSq.r - mean.r * mean.r) +
			(meanSq.g - mean.g * mean.g) +
			(meanSq.b - mean.b * mean.b)) / 3.0f;
		return variance;
	}

	// Smooth variance values
	float smoothVariance(Film* film, int startX, int startY, int endX, int endY)
	{
		float totalVariance = 0.0f;
		int numBlocks = 0;
		int smoothRadius = 2;

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
		bool usePathTracing = false;
		bool useLightTracing = false;
		bool useInstantRadiosity = true;

		film->incrementSPP();

		const int N_VPLs = 10000;

		// Instant Radiosity
		if (useInstantRadiosity)
		{
			// Generate VPLs
			auto startVPL = std::chrono::high_resolution_clock::now();
			traceVPLs(N_VPLs); // Parallelized
			auto endVPL = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> vplTime = endVPL - startVPL;
			// Check if VPLs were actually generated
			if (scene->vplList.empty() && N_VPLs > 0 && !scene->lights.empty()) {
				std::cerr << "Error: No VPLs generated after traceVPLs finished. Rendering will likely be black or direct only." << std::endl;
			}

			// Gather illumination at each pixel
			std::cout << "Starting Gathering Pass..." << std::endl;
			auto startGather = std::chrono::high_resolution_clock::now();
			const int tileSize = 16;
			const int tileCountX = (film->width + tileSize - 1) / tileSize;
			const int tileCountY = (film->height + tileSize - 1) / tileSize;
			std::atomic<int> nextTileIndex(0);

			std::vector<std::thread> workers(numProcs);
			for (int i = 0; i < numProcs; ++i) {
				workers[i] = std::thread([&, i]() {
					Sampler* sampler = &samplers[i];
					while (true) {
						int tileIdx = nextTileIndex.fetch_add(1);
						if (tileIdx >= tileCountX * tileCountY) break;

						int tileX = tileIdx % tileCountX;
						int tileY = tileIdx / tileCountX;
						int startX = tileX * tileSize;
						int startY = tileY * tileSize;
						int endX = min(startX + tileSize, (int)film->width);
						int endY = min(startY + tileSize, (int)film->height);

						for (int y = startY; y < endY; ++y) {
							for (int x = startX; x < endX; ++x) {
								// Generate camera ray
								float px = (float)x + sampler->next();
								float py = (float)y + sampler->next();
								Ray cameraRay = scene->camera.generateRay(px, py);

								// Trace camera ray to find first hit
								IntersectionData isect = scene->traverse(cameraRay);
								Colour pixelColour(0.0f, 0.0f, 0.0f);

								if (isect.t == FLT_MAX) {
									if (scene->background) {
										pixelColour = scene->background->evaluate(cameraRay.dir);
									}
								}
								else {
									ShadingData sData = scene->calculateShadingData(isect, cameraRay);

									// Check if BSDF is valid before proceeding
									if (!sData.bsdf) {
										pixelColour = Colour(1.0f, 0.0f, 1.0f);
									}
									else {
										// Add emission if the first hit is a light source
										if (sData.bsdf->isLight()) {
											pixelColour += sData.bsdf->emit(sData, sData.wo);
										}

										// Calculate Direct Lighting
										Colour L_direct = computeDirect(sData, sampler);

										// Calculate Indirect Lighting using VPLs
										Colour L_indirect = computeIndirectVPL(sData, N_VPLs);

										pixelColour += L_direct + L_indirect;
										if (!pixelColour.isValid() || pixelColour.r < 0 || pixelColour.g < 0 || pixelColour.b < 0) {
											pixelColour = Colour(0.0f, 0.0f, 0.0f);
										}
									}
								}
								film->splat(px, py, pixelColour);
							}
						}
					}
					});
			}

			for (int i = 0; i < numProcs; ++i) {
				if (workers[i].joinable()) {
					workers[i].join();
				}
			}
			auto endGather = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> gatherTime = endGather - startGather;
			std::cout << "Gathering Pass Time: " << gatherTime.count() << " ms" << std::endl;

			// Tonemap and Draw
			for (int y = 0; y < film->height; ++y) {
				for (int x = 0; x < film->width; ++x) {
					unsigned char r, g, b;
					film->tonemap(x, y, r, g, b);
					canvas->draw(x, y, r, g, b);
				}
			}
		}
		// Path Tracing
		else if (usePathTracing)
		{
			std::cout << "Using Path Tracing (MIS)..." << std::endl;
			auto startGather = std::chrono::high_resolution_clock::now();
			const int tileSize = 16;
			const int tileCountX = (film->width + tileSize - 1) / tileSize;
			const int tileCountY = (film->height + tileSize - 1) / tileSize;
			std::atomic<int> nextTileIndex(0);

			std::vector<std::thread> workers(numProcs);
			for (int i = 0; i < numProcs; ++i) {
				workers[i] = std::thread([&, i]() {
					Sampler* sampler = &samplers[i];
					while (true) {
						int tileIdx = nextTileIndex.fetch_add(1);
						if (tileIdx >= tileCountX * tileCountY) break;

						int tileX = tileIdx % tileCountX;
						int tileY = tileIdx / tileCountX;
						int startX = tileX * tileSize;
						int startY = tileY * tileSize;
						int endX = min(startX + tileSize, (int)film->width);
						int endY = min(startY + tileSize, (int)film->height);

						for (int y = startY; y < endY; ++y) {
							for (int x = startX; x < endX; ++x) {
								float px = (float)x + sampler->next();
								float py = (float)y + sampler->next();
								Ray ray = scene->camera.generateRay(px, py);
								Colour throughput(1.0f, 1.0f, 1.0f);
								Colour col = pathTraceMIS(ray, throughput, 0, sampler);

								if (!col.isValid()) col = Colour(0, 0, 0);

								film->splat(px, py, col);

							}
						}
					}
					});
			}
			for (int i = 0; i < numProcs; ++i) {
				if (workers[i].joinable()) workers[i].join();
			}
			auto endGather = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> gatherTime = endGather - startGather;
			std::cout << "Path Tracing Pass Time: " << gatherTime.count() << " ms" << std::endl;

			// Draw after all PT threads finish
			for (int y = 0; y < film->height; ++y) {
				for (int x = 0; x < film->width; ++x) {
					unsigned char r, g, b;
					film->tonemap(x, y, r, g, b);
					canvas->draw(x, y, r, g, b);
				}
			}
		}
		else if (useLightTracing)
		{
			std::cout << "Using Light Tracing..." << std::endl;
			auto startGather = std::chrono::high_resolution_clock::now();
			const int pathsPerThreadBatch = 10000;
			const int numBatches = 100;
			std::atomic<int> nextBatchIndex(0);

			std::vector<std::thread> workers(numProcs);
			for (int i = 0; i < numProcs; ++i) {
				workers[i] = std::thread([&, i]() {
					Sampler* sampler = &samplers[i];
					while (true) {
						int batchIdx = nextBatchIndex.fetch_add(1);
						if (batchIdx >= numBatches) break;

						for (int p = 0; p < pathsPerThreadBatch; ++p) {
							lightTrace(sampler);
						}
					}
					});
			}
			for (int i = 0; i < numProcs; ++i) {
				if (workers[i].joinable()) workers[i].join();
			}
			auto endGather = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double, std::milli> gatherTime = endGather - startGather;
			for (int y = 0; y < film->height; ++y) {
				for (int x = 0; x < film->width; ++x) {
					unsigned char r, g, b;
					film->tonemap(x, y, r, g, b);
					canvas->draw(x, y, r, g, b);
				}
			}
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


