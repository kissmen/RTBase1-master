#pragma once

#include "Core.h"
#include "Sampling.h"
#include "Geometry.h"
#include "Imaging.h"
#include "Materials.h"
#include "Lights.h"
#include "Scene.h"
#include "VPL.h"

class Camera
{
public:
	Matrix projectionMatrix;
	Matrix inverseProjectionMatrix;
	Matrix camera;
	Matrix cameraToView;
	float width = 0;
	float height = 0;
	Vec3 origin;
	Vec3 viewDirection;
	float Afilm;
	void init(Matrix ProjectionMatrix, int screenwidth, int screenheight)
	{
		projectionMatrix = ProjectionMatrix;
		inverseProjectionMatrix = ProjectionMatrix.invert();
		width = (float)screenwidth;
		height = (float)screenheight;
		float Wlens = (2.0f / ProjectionMatrix.a[1][1]);
		float aspect = ProjectionMatrix.a[0][0] / ProjectionMatrix.a[1][1];
		float Hlens = Wlens * aspect;
		Afilm = Wlens * Hlens;
	}
	void updateView(Matrix V)
	{
		camera = V;
		cameraToView = V.invert();
		origin = camera.mulPoint(Vec3(0, 0, 0));
		viewDirection = inverseProjectionMatrix.mulPointAndPerspectiveDivide(Vec3(0, 0, 1));
		viewDirection = camera.mulVec(viewDirection);
		viewDirection = viewDirection.normalize();
	}
	// Add code here (done)
	Ray generateRay(float x, float y)
	{
		float xprime = x / width;
		float yprime = 1.0f - (y / height);
		xprime = (xprime * 2.0f) - 1.0f;
		yprime = (yprime * 2.0f) - 1.0f;
		Vec3 dir(xprime, yprime, 1.0f);
		dir = inverseProjectionMatrix.mulPoint(dir);
		dir = camera.mulVec(dir);
		dir = dir.normalize();
		// Print ray direction
		//std::cout << "Ray Direction: " << dir.x << ", " << dir.y << ", " << dir.z << std::endl;
		return Ray(origin, dir);
	}
	bool projectOntoCamera(const Vec3& p, float& x, float& y)
	{
		Vec3 pview = cameraToView.mulPoint(p);
		Vec3 pproj = projectionMatrix.mulPointAndPerspectiveDivide(pview);
		std::cout << "Projected Coordinates: (" << pproj.x << ", " << pproj.y << ", " << pproj.z << ")" << std::endl;
		x = (pproj.x + 1.0f) * 0.5f;
		y = (pproj.y + 1.0f) * 0.5f;
		if (x < 0 || x > 1.0f || y < 0 || y > 1.0f)
		{
			return false;
		}
		x = x * width;
		y = 1.0f - y;
		y = y * height;
		return true;
	}
};

class Scene
{
public:
	std::vector<Triangle> triangles;
	std::vector<BSDF*> materials;
	std::vector<Light*> lights;
	std::vector<VPL> vplList;
	Light* background = NULL;
	BVHNode* bvh = NULL;
	Camera camera;
	AABB bounds;
	std::vector<float> lightCDF;
	float totalLightPower = 0.0f;
	bool builtLightCDF = false;

	void build()
	{
		// Add BVH building code here
		vector<int> indices;
		for (int i = 0; i < triangles.size(); ++i) {
			if (!triangles[i].vertices[0].p.isValid() || !triangles[i].vertices[1].p.isValid() || !triangles[i].vertices[2].p.isValid()) {

			}
			AABB triangleBounds(triangles[i]);

			// Extend the bounds with the min and max of the AABB
			bounds.extend(triangleBounds.min);  // Use triangleBounds.min
			bounds.extend(triangleBounds.max);  // Use triangleBounds.max
		}
		bvh = new BVHNode();
		//bvh->build(triangles);
		bvh->buildWithSAH(triangles);
		//bvh->buildWithMortonSAH(triangles);
		//bvh->buildWithGridSAHParallel(triangles);

		// Do not touch the code below this line!
		// Build light list
		for (int i = 0; i < triangles.size(); i++)
		{
			if (materials[triangles[i].materialIndex]->isLight())
			{
				AreaLight* light = new AreaLight();
				light->triangle = &triangles[i];
				light->emission = materials[triangles[i].materialIndex]->emission;
				//light->emittingNormal = -triangles[i].n;
				light->emittingNormal = triangles[i].gNormal();
				std::cout << "Light Triangle ID: " << triangles[i].id
					<< " gNormal: " << triangles[i].gNormal()
					<< " Assigned Emitting Normal: " << light->emittingNormal << std::endl;
				lights.push_back(light);
			}
		}
	}

	void removeInvalidTriangles() {
		std::vector<Triangle> validTriangles;
		for (auto& triangle : triangles) {
			if (triangle.vertices[0].p.isValid() && triangle.vertices[1].p.isValid() && triangle.vertices[2].p.isValid()) {
				validTriangles.push_back(triangle);
			}
		}
		triangles = validTriangles;
	}
	

	// BVH traversal
	IntersectionData traverse(const Ray& ray) {
		return bvh->traverse(ray, triangles);
	}

	Light* sampleLight(Sampler* sampler, float& pmf)
	{
		// Select light source and calculate its probability mass function
		/*if (lights.empty()) return nullptr;
		float r1 = sampler->next();
		pmf = 1.0f / (float)lights.size();
		int index = std::min((int)(r1 * lights.size()), (int)(lights.size() - 1));
		return lights[index];*/
		static bool cdfBuiltOnce = false;
		if (!builtLightCDF) {
			buildLightPowerCDF();
			builtLightCDF = true;
			cdfBuiltOnce = true;
		}
		if (lights.empty()) {
			pmf = 1.0f;
			std::cerr << "[Warning] No lights available in scene! Returning nullptr.\n";
			return nullptr;
		}

		float r = sampler->next();
		int index = binarySearchLightCDF(r);

		float power_i = (index == 0) ? (lightCDF[0] * totalLightPower)
			: ((lightCDF[index] - lightCDF[index - 1]) * totalLightPower);
		pmf = power_i / totalLightPower;

		return lights[index];
	}
	void buildLightPowerCDF()
	{
		lightCDF.resize(lights.size());
		float accum = 0.0f;
		for (int i = 0; i < lights.size(); i++) {
			float pwr = lights[i]->totalIntegratedPower();
			accum += pwr;
			lightCDF[i] = accum;
		}
		totalLightPower = accum;
		for (int i = 0; i < lights.size(); i++) {
			lightCDF[i] /= totalLightPower;
		}
	}

	int binarySearchLightCDF(float r)
	{
		int left = 0;
		int right = (int)lightCDF.size() - 1;
		while (left < right)
		{
			int mid = (left + right) >> 1;
			if (lightCDF[mid] < r)
				left = mid + 1;
			else
				right = mid;
		}
		return left;
	}

	Light* findLightByTriangleID(int triID) {
		for (auto* L : lights) {
			if (L->isArea()) {
				AreaLight* A = static_cast<AreaLight*>(L);
				if (A->triangle && A->triangle->id == triID) {
					return A;
				}
			}
		}
		return nullptr;
	}

	// Do not modify any code below this line
	void init(std::vector<Triangle> meshTriangles, std::vector<BSDF*> meshMaterials, Light* _background)
	{
		for (int i = 0; i < meshTriangles.size(); i++)
		{
			triangles.push_back(meshTriangles[i]);
			bounds.extend(meshTriangles[i].vertices[0].p);
			bounds.extend(meshTriangles[i].vertices[1].p);
			bounds.extend(meshTriangles[i].vertices[2].p);
		}
		for (int i = 0; i < meshMaterials.size(); i++)
		{
			materials.push_back(meshMaterials[i]);
		}
		background = _background;
		if (_background->totalIntegratedPower() > 0)
		{
			lights.push_back(_background);
		}
	}
	bool visible(const Vec3& p1, const Vec3& p2)
	{
		Ray ray;
		Vec3 dir = p2 - p1;
		float maxT = dir.length() - (2.0f * EPSILON);
		dir = dir.normalize();
		ray.init(p1 + (dir * EPSILON), dir);
		return bvh->traverseVisible(ray, triangles, maxT);
	}
	Colour emit(Triangle* light, ShadingData shadingData, Vec3 wi)
	{
		return materials[light->materialIndex]->emit(shadingData, wi);
	}
	ShadingData calculateShadingData(IntersectionData intersection, Ray& ray)
	{
		ShadingData shadingData = {};
		if (intersection.t < FLT_MAX)
		{
			shadingData.x = ray.at(intersection.t);
			shadingData.gNormal = triangles[intersection.ID].gNormal();
			triangles[intersection.ID].interpolateAttributes(intersection.alpha, intersection.beta, intersection.gamma, shadingData.sNormal, shadingData.tu, shadingData.tv);
			shadingData.bsdf = materials[triangles[intersection.ID].materialIndex];
			shadingData.wo = -ray.dir;
			if (shadingData.bsdf->isTwoSided())
			{
				if (Dot(shadingData.wo, shadingData.sNormal) < 0)
				{
					shadingData.sNormal = -shadingData.sNormal;
				}
				if (Dot(shadingData.wo, shadingData.gNormal) < 0)
				{
					shadingData.gNormal = -shadingData.gNormal;
				}
			}
			shadingData.frame.fromVector(shadingData.sNormal);
			shadingData.t = intersection.t;
		} else
		{
			shadingData.wo = -ray.dir;
			shadingData.t = intersection.t;
		}
		return shadingData;
	}
};