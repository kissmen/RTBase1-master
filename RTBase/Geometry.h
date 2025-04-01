#pragma once

#include <algorithm>   // for std::sort, std::swap, etc.
#include <functional>  // for std::function
#include <vector>      // for std::vector
#include <cmath>       // for sqrt, fabs
#include <cfloat>      // for FLT_MAX
#include <future> 

#include "Core.h"
#include "Sampling.h"

#define EPSILON 0.001f

uint32_t floatToBits(float f);
uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z);

using namespace std;

class Ray
{
public:
	Vec3 o;
	Vec3 dir;
	Vec3 invDir;
	Ray()
	{
	}
	Ray(Vec3 _o, Vec3 _d)
	{
		init(_o, _d);
	}
	void init(Vec3 _o, Vec3 _d)
	{
		o = _o;
		dir = _d;
		invDir = Vec3(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
	}
	Vec3 at(const float t) const
	{
		return (o + (dir * t));
	}
};

class Plane
{
public:
	Vec3 n;
	float d;
	void init(Vec3& _n, float _d)
	{
		n = _n;
		d = _d;
	}
	// Add code here
	bool rayIntersect(Ray& r, float& t)
	{
		// n·dir
		float denom = Dot(n, r.dir);

		// Determine if parallel to plane
		if (fabs(denom) < EPSILON)
		{
			// Parallel, not intersecting
			return false;
		}
		// t
		float numerator = d - Dot(n, r.o);
		float tCandidate = numerator / denom;

		// If t>0, ray forward
		if (tCandidate > EPSILON)
		{
			t = tCandidate;
			return true;
		}

		return false;
	}
};

class Triangle
{
public:
	Vertex vertices[3];
	unsigned int id;
	Vec3 e1; // Edge 1
	Vec3 e2; // Edge 2
	Vec3 n; // Geometric Normal
	float area; // Triangle area
	float d; // For ray triangle if needed
	unsigned int materialIndex;
	mutable bool isValid = false;
	void init(Vertex v0, Vertex v1, Vertex v2, unsigned int _materialIndex, unsigned int _id = 0)
	{
		materialIndex = _materialIndex;
		id = _id;
		vertices[0] = v0;
		vertices[1] = v1;
		vertices[2] = v2;
		e1 = vertices[2].p - vertices[1].p;
		e2 = vertices[0].p - vertices[2].p;
		n = e1.cross(e2).normalize();
		area = e1.cross(e2).length() * 0.5f;
		d = Dot(n, vertices[0].p);
		isValid = true;
	}
	/*Vec3 centre() const
	{
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}*/

	Vec3 centre() const
	{
		if (!isValid || !vertices[0].p.isValid() || !vertices[1].p.isValid() || !vertices[2].p.isValid()) {
			return Vec3(0.0f, 0.0f, 0.0f);
		}
		return (vertices[0].p + vertices[1].p + vertices[2].p) / 3.0f;
	}


	// Add code here (done)
	bool rayIntersect(const Ray& r, float& t, float& u, float& v) const
	{
		float denom = Dot(n, r.dir);
		if (denom == 0) { return false; }
		t = (d - Dot(n, r.o)) / denom;
		if (t < 0) { return false; }
		Vec3 p = r.at(t);
		float invArea = 1.0f / Dot(e1.cross(e2), n);
		u = Dot(e1.cross(p - vertices[1].p), n) * invArea;
		if (u < 0 || u > 1.0f) { return false; }
		v = Dot(e2.cross(p - vertices[2].p), n) * invArea;
		if (v < 0 || (u + v) > 1.0f) { return false; }
		return true;
	}

	void interpolateAttributes(const float alpha, const float beta, const float gamma, Vec3& interpolatedNormal, float& interpolatedU, float& interpolatedV) const
	{
		interpolatedNormal = vertices[0].normal * alpha + vertices[1].normal * beta + vertices[2].normal * gamma;
		interpolatedNormal = interpolatedNormal.normalize();
		interpolatedU = vertices[0].u * alpha + vertices[1].u * beta + vertices[2].u * gamma;
		interpolatedV = vertices[0].v * alpha + vertices[1].v * beta + vertices[2].v * gamma;
	}

	// Add code here (done)
	Vec3 sample(Sampler* sampler, float& pdf)
	{
		float r1 = sampler->next();
		float r2 = sampler->next();

		// α = 1 - sqrt(r1), β = r2 * sqrt(r1), γ = 1 - α - β
		float sqrtR1 = sqrt(r1);
		float alpha = 1.0f - sqrtR1;
		float beta = r2 * sqrtR1;
		float gamma = 1.0f - alpha - beta;

		// Linear combination of 3 vertices with center of gravity coordinates
		Vec3 p = vertices[0].p * alpha + vertices[1].p * beta + vertices[2].p * gamma;

		// When sampling triangles uniformly, pdf = 1 / area
		pdf = 1.0f / area;

		return p;
	}
	Vec3 gNormal()
	{
		return (n * (Dot(vertices[0].normal, n) > 0 ? 1.0f : -1.0f));
	}
};

class AABB
{
public:
	Vec3 max;
	Vec3 min;
	AABB()
	{
		reset();
	}
	AABB(const Triangle& t)
	{
		reset();
		extend(t.vertices[0].p);
		extend(t.vertices[1].p);
		extend(t.vertices[2].p);
	}
	void reset()
	{
		max = Vec3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		min = Vec3(FLT_MAX, FLT_MAX, FLT_MAX);
	}
	void extend(const Vec3 p)
	{
		max = Max(max, p);
		min = Min(min, p);
	}
	// Add code here
	bool rayAABB(const Ray& r, float& tMin, float& tMax) const
	{
		for (int i = 0; i < 3; i++) {
			if (r.dir[i] == 0.0f) {
				if (r.o[i] < min[i] || r.o[i] > max[i]) {
					return false;
				}
			}
			else {
				float t0 = (min[i] - r.o[i]) * r.invDir[i];
				float t1 = (max[i] - r.o[i]) * r.invDir[i];
				if (t0 > t1) std::swap(t0, t1);
				tMin = std::max(tMin, t0);
				tMax = std::min(tMax, t1);
				if (tMax < tMin)
					return false;
			}
		}
		return true;
	}
	// Add code here
	bool rayAABB(const Ray& r) const
	{
		float tMin = EPSILON;
		float tMax = FLT_MAX;
		return rayAABB(r, tMin, tMax);
	}
	// Add code here
	float area() const {
		Vec3 size = max - min;
		return 2.0f * (size.x * size.y + size.y * size.z + size.x * size.z);
	}
};

class Sphere
{
public:
	Vec3 centre;
	float radius;
	void init(Vec3& _centre, float _radius)
	{
		centre = _centre;
		radius = _radius;
	}
	// Add code here (done)
	bool rayIntersect(Ray& r, float& t)
	{
		Vec3 oc = r.o - centre;
		float a = Dot(r.dir, r.dir);
		float b = 2.0f * Dot(oc, r.dir);
		float c = Dot(oc, oc) - radius * radius;
		float discriminant = b * b - 4 * a * c;
		if (discriminant < 0)
			return false;
		float sqrtDisc = sqrtf(discriminant);
		float t0 = (-b - sqrtDisc) / (2 * a);
		float t1 = (-b + sqrtDisc) / (2 * a);
		if (t0 > EPSILON) {
			t = t0;
			return true;
		}
		if (t1 > EPSILON) {
			t = t1;
			return true;
		}
		return false;
	}
};

struct IntersectionData
{
	unsigned int ID;
	float t;
	float alpha;
	float beta;
	float gamma;
};

#define MAXNODE_TRIANGLES 8
#define TRAVERSE_COST 1.0f
#define TRIANGLE_COST 2.0f
#define BUILD_BINS 16
#define PARALLEL_THRESHOLD 1000

class BVHNode
{
public:
	AABB bounds;
	BVHNode* r;
	BVHNode* l;
	vector<int> triangleIndices;
	// This can store an offset and number of triangles in a global triangle list for example
	// But you can store this however you want!
	// unsigned int offset;
	// unsigned char num;
	BVHNode()
	{
		r = NULL;
		l = NULL;
	}

	// Note there are several options for how to implement the build method. Update this as required
	void build(vector<Triangle>& inputTriangles)
	{
		// Add BVH building code here
		vector<int> indices(inputTriangles.size());
		for (int i = 0; i < inputTriangles.size(); i++)
		{
			indices[i] = i;
		}

		std::function<void(BVHNode*, vector<int>&)> buildFunc = [&](BVHNode* node, vector<int>& idx) {
			node->bounds.reset();
			for (int i : idx)
			{
				node->bounds.extend(inputTriangles[i].vertices[0].p);
				node->bounds.extend(inputTriangles[i].vertices[1].p);
				node->bounds.extend(inputTriangles[i].vertices[2].p);
			}

			if (idx.size() <= MAXNODE_TRIANGLES)
			{
				node->triangleIndices = idx;
				return;
			}
			Vec3 extent = node->bounds.max - node->bounds.min;
			int axis = 0;
			if (extent.y > extent.x && extent.y > extent.z)
				axis = 1;
			else if (extent.z > extent.x && extent.z > extent.y)
				axis = 2;
			std::sort(idx.begin(), idx.end(), [&](int a, int b) {
				Vec3 ca = inputTriangles[a].centre();
				Vec3 cb = inputTriangles[b].centre();
				if (axis == 0) return ca.x < cb.x;
				else if (axis == 1) return ca.y < cb.y;
				else return ca.z < cb.z;
				});
			int mid = idx.size() / 2;
			vector<int> leftIndices(idx.begin(), idx.begin() + mid);
			vector<int> rightIndices(idx.begin() + mid, idx.end());

			node->l = new BVHNode();
			node->r = new BVHNode();
			buildFunc(node->l, leftIndices);
			buildFunc(node->r, rightIndices);
			};

		buildFunc(this, indices);
	}

	// Optimize BVH via surface area heuristic
	void buildWithSAH(vector<Triangle>& inputTriangles)
	{
		vector<int> indices(inputTriangles.size());
		for (int i = 0; i < inputTriangles.size(); i++)
		{
			indices[i] = i;
		}

		std::function<void(BVHNode*, vector<int>&)> buildFunc = [&](BVHNode* node, vector<int>& idx) {
			// Calculate the enclosing box of current node
			node->bounds.reset();
			for (int i : idx)
			{
				node->bounds.extend(inputTriangles[i].vertices[0].p);
				node->bounds.extend(inputTriangles[i].vertices[1].p);
				node->bounds.extend(inputTriangles[i].vertices[2].p);
			}

			// If triangles number is small, the index is stored as a leaf node
			if (idx.size() <= MAXNODE_TRIANGLES)
			{
				node->triangleIndices = idx;
				return;
			}

			// Selecting Segment Axis and Segment Position with SAH
			float bestCost = FLT_MAX;
			int bestAxis = 0;
			int bestSplit = 0;

			for (int axis = 0; axis < 3; axis++)
			{
				// Triangle by axis
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
					Vec3 ca = inputTriangles[a].centre();
					Vec3 cb = inputTriangles[b].centre();
					if (axis == 0) return ca.x < cb.x;
					else if (axis == 1) return ca.y < cb.y;
					else return ca.z < cb.z;
					});

				// Compute AABB and cost of prefixes and suffixes
				vector<AABB> leftBounds(idx.size());
				vector<AABB> rightBounds(idx.size());
				leftBounds[0] = AABB(inputTriangles[idx[0]]);
				for (int i = 1; i < idx.size(); i++)
				{
					leftBounds[i] = leftBounds[i - 1];
					leftBounds[i].extend(inputTriangles[idx[i]].vertices[0].p);
					leftBounds[i].extend(inputTriangles[idx[i]].vertices[1].p);
					leftBounds[i].extend(inputTriangles[idx[i]].vertices[2].p);
				}

				rightBounds[idx.size() - 1] = AABB(inputTriangles[idx[idx.size() - 1]]);
				for (int i = idx.size() - 2; i >= 0; i--)
				{
					rightBounds[i] = rightBounds[i + 1];
					rightBounds[i].extend(inputTriangles[idx[i]].vertices[0].p);
					rightBounds[i].extend(inputTriangles[idx[i]].vertices[1].p);
					rightBounds[i].extend(inputTriangles[idx[i]].vertices[2].p);
				}

				// Iterate over all split points and compute SAH cost
				for (int i = 0; i < idx.size() - 1; i++)
				{
					float leftArea = leftBounds[i].area();
					float rightArea = rightBounds[i + 1].area();
					float leftCost = leftArea * (i + 1);
					float rightCost = rightArea * (idx.size() - (i + 1));
					float totalCost = leftCost + rightCost;

					if (totalCost < bestCost)
					{
						bestCost = totalCost;
						bestAxis = axis;
						bestSplit = i;
					}
				}
			}

			// Sorting according best axis
			if (bestAxis == 0)
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
				return inputTriangles[a].centre().x < inputTriangles[b].centre().x;
					});
			else if (bestAxis == 1)
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
				return inputTriangles[a].centre().y < inputTriangles[b].centre().y;
					});
			else
				std::sort(idx.begin(), idx.end(), [&](int a, int b) {
				return inputTriangles[a].centre().z < inputTriangles[b].centre().z;
					});

			vector<int> leftIndices(idx.begin(), idx.begin() + bestSplit + 1);
			vector<int> rightIndices(idx.begin() + bestSplit + 1, idx.end());

			// Creating left and right subtrees
			node->l = new BVHNode();
			node->r = new BVHNode();

			// Recursive of left and right subtrees
			buildFunc(node->l, leftIndices);
			buildFunc(node->r, rightIndices);
			};
		buildFunc(this, indices);
	}

	void traverse(const Ray& ray, const vector<Triangle>& triangles, IntersectionData& intersection)
	{
		// Add BVH Traversal code here
		float tMin = EPSILON;
		float tMax = intersection.t;
		if (!bounds.rayAABB(ray, tMin, tMax))
			return;
		if (!l && !r)
		{
			for (int idx : triangleIndices)
			{
				float t, u, v;
				if (triangles[idx].rayIntersect(ray, t, u, v))
				{
					if (t < intersection.t)
					{
						intersection.t = t;
						intersection.ID = idx;
						intersection.alpha = u;
						intersection.beta = v;
						intersection.gamma = 1.0f - (u + v);
					}
				}
			}
			return;
		}
		if (l)
			l->traverse(ray, triangles, intersection);
		if (r)
			r->traverse(ray, triangles, intersection);
	}

	IntersectionData traverse(const Ray& ray, const vector<Triangle>& triangles)
	{
		IntersectionData intersection;
		intersection.t = FLT_MAX;
		traverse(ray, triangles, intersection);
		return intersection;
	}

	bool traverseVisible(const Ray& ray, const vector<Triangle>& triangles, const float maxT) {
		float tMin = EPSILON;
		float tMax = maxT;

		if (!bounds.rayAABB(ray, tMin, tMax)) {
			float nodeTMin = EPSILON;
			float nodeTMax = maxT;
			if (!bounds.rayAABB(ray, nodeTMin, nodeTMax)) {
				return true;
			}
		}
		if (!l && !r) { // Leaf node
			for (int idx : triangleIndices) {
				float t, u, v;
				// Check intersection with specific triangle
				if (triangles[idx].rayIntersect(ray, t, u, v)) {
					if (t > EPSILON && t < maxT) {
						return false;
					}
				}
			}
			return true;
		}
		// Internal node Recurse
		bool visible = true;
		if (l) {
			// Pass the original maxT down
			visible = l->traverseVisible(ray, triangles, maxT);
		}
		if (visible && r) {
			visible = r->traverseVisible(ray, triangles, maxT);
		}
		return visible; // Return true only if NO occlusion was found in relevant children
	}
};

// Convert a float in [0,1] to a 10-bit integer.
inline uint32_t floatTo10Bit(float f)
{
	f = std::min(std::max(f, 0.0f), 1.0f);
	return static_cast<uint32_t>(f * 1023.0f);
}

// Interleave 10 bits of x, y, and z into a 30-bit Morton code.
inline uint64_t morton3D(uint32_t x, uint32_t y, uint32_t z)
{
	auto splitBy3 = [](uint32_t a) -> uint64_t {
		uint64_t x = a & 0x3ff; // 10 bits
		x = (x | (x << 16)) & 0x30000ff;
		x = (x | (x << 8)) & 0x300f00f;
		x = (x | (x << 4)) & 0x30c30c3;
		x = (x | (x << 2)) & 0x9249249;
		return x;
		};
	return (splitBy3(x) << 2) | (splitBy3(y) << 1) | splitBy3(z);
}

/*
void buildWithMortonSAH(vector<Triangle>& inputTriangles)
{
	// 1. Compute scene bounds
	AABB sceneBounds;
	sceneBounds.reset();
	for (int i = 0; i < inputTriangles.size(); i++) {
		sceneBounds.extend(inputTriangles[i].vertices[0].p);
		sceneBounds.extend(inputTriangles[i].vertices[1].p);
		sceneBounds.extend(inputTriangles[i].vertices[2].p);
	}

	// 2. Create an index array and compute Morton codes
	int nTriangles = inputTriangles.size();
	vector<int> indices(nTriangles);
	vector<uint64_t> mortonCodes(nTriangles);
	for (int i = 0; i < nTriangles; i++) {
		indices[i] = i;
		Vec3 center = inputTriangles[i].centre();
		// Normalize center to [0, 1]
		float nx = (center.x - sceneBounds.min.x) / (sceneBounds.max.x - sceneBounds.min.x);
		float ny = (center.y - sceneBounds.min.y) / (sceneBounds.max.y - sceneBounds.min.y);
		float nz = (center.z - sceneBounds.min.z) / (sceneBounds.max.z - sceneBounds.min.z);
		// Map to 10-bit integers (0-1023)
		uint32_t ix = static_cast<uint32_t>(std::min(std::max(nx * 1023.0f, 0.0f), 1023.0f));
		uint32_t iy = static_cast<uint32_t>(std::min(std::max(ny * 1023.0f, 0.0f), 1023.0f));
		uint32_t iz = static_cast<uint32_t>(std::min(std::max(nz * 1023.0f, 0.0f), 1023.0f));
		mortonCodes[i] = morton3D(ix, iy, iz);
	}

	// 3. Sort indices by Morton codes
	std::sort(indices.begin(), indices.end(), [&](int a, int b) {
		return mortonCodes[a] < mortonCodes[b];
		});

	// 4. Recursively build the BVH using SAH over the sorted indices.
	// We add parallelism if the number of triangles is above a threshold.
	std::function<void(BVHNode*, int, int)> buildRec = [&](BVHNode* node, int start, int end) {
		node->bounds.reset();
		for (int i = start; i <= end; i++) {
			int idx = indices[i];
			node->bounds.extend(inputTriangles[idx].vertices[0].p);
			node->bounds.extend(inputTriangles[idx].vertices[1].p);
			node->bounds.extend(inputTriangles[idx].vertices[2].p);
		}
		int count = end - start + 1;
		if (count <= MAXNODE_TRIANGLES) {
			node->triangleIndices.assign(indices.begin() + start, indices.begin() + end + 1);
			return;
		}
		// Compute prefix and suffix AABBs
		vector<AABB> prefix(count);
		vector<AABB> suffix(count);
		prefix[0] = AABB(inputTriangles[indices[start]]);
		for (int i = 1; i < count; i++) {
			int idx = indices[start + i];
			prefix[i] = prefix[i - 1];
			prefix[i].extend(inputTriangles[idx].vertices[0].p);
			prefix[i].extend(inputTriangles[idx].vertices[1].p);
			prefix[i].extend(inputTriangles[idx].vertices[2].p);
		}
		suffix[count - 1] = AABB(inputTriangles[indices[end]]);
		for (int i = count - 2; i >= 0; i--) {
			int idx = indices[start + i];
			suffix[i] = suffix[i + 1];
			suffix[i].extend(inputTriangles[idx].vertices[0].p);
			suffix[i].extend(inputTriangles[idx].vertices[1].p);
			suffix[i].extend(inputTriangles[idx].vertices[2].p);
		}
		float bestCost = FLT_MAX;
		int bestSplit = start;
		for (int i = 0; i < count - 1; i++) {
			float leftCost = prefix[i].area() * (i + 1);
			float rightCost = suffix[i + 1].area() * (count - i - 1);
			float totalCost = leftCost + rightCost;
			if (totalCost < bestCost) {
				bestCost = totalCost;
				bestSplit = start + i;
			}
		}
		node->l = new BVHNode();
		node->r = new BVHNode();
		// Parallelize if the subproblem is large
		if (count > PARALLEL_THRESHOLD) {
			auto futureLeft = std::async(std::launch::async, buildRec, node->l, start, bestSplit);
			auto futureRight = std::async(std::launch::async, buildRec, node->r, bestSplit + 1, end);
			futureLeft.wait();
			futureRight.wait();
		}
		else {
			buildRec(node->l, start, bestSplit);
			buildRec(node->r, bestSplit + 1, end);
		}
		};

	// Create a temporary root and build recursively.
	BVHNode* tempRoot = new BVHNode();
	buildRec(tempRoot, 0, nTriangles - 1);
	*this = *tempRoot;
	delete tempRoot;
}

// In BVHNode class – add a new build method:
void buildWithGridSAHParallel(std::vector<Triangle>& inputTriangles)
{
	// STEP 1: Compute the scene bounding box.
	AABB sceneBounds;
	sceneBounds.reset();
	int nTriangles = inputTriangles.size();
	for (int i = 0; i < nTriangles; i++) {
		sceneBounds.extend(inputTriangles[i].vertices[0].p);
		sceneBounds.extend(inputTriangles[i].vertices[1].p);
		sceneBounds.extend(inputTriangles[i].vertices[2].p);
	}

	// STEP 2: Choose a grid resolution (here we use a constant resolution per axis).
	const int GRID_RES = 64; // You may adjust this resolution based on scene complexity.

	// STEP 3: Compute a cell index for each triangle based on its centroid.
	// We'll use a 1D index computed from 3D cell coordinates.
	std::vector<int> indices(nTriangles);
	std::vector<uint32_t> cellCodes(nTriangles);
	for (int i = 0; i < nTriangles; i++) {
		indices[i] = i;
		Vec3 center = inputTriangles[i].centre();
		// Normalize center to [0, 1]
		float nx = (center.x - sceneBounds.min.x) / (sceneBounds.max.x - sceneBounds.min.x);
		float ny = (center.y - sceneBounds.min.y) / (sceneBounds.max.y - sceneBounds.min.y);
		float nz = (center.z - sceneBounds.min.z) / (sceneBounds.max.z - sceneBounds.min.z);
		// Clamp slightly to avoid exactly 1.0
		nx = std::min(std::max(nx, 0.0f), 0.9999f);
		ny = std::min(std::max(ny, 0.0f), 0.9999f);
		nz = std::min(std::max(nz, 0.0f), 0.9999f);
		// Compute discrete cell coordinates (0 .. GRID_RES-1)
		int cx = static_cast<int>(nx * GRID_RES);
		int cy = static_cast<int>(ny * GRID_RES);
		int cz = static_cast<int>(nz * GRID_RES);
		// Compute a 1D cell code (assuming row‐major order: x changes fastest)
		cellCodes[i] = cx + cy * GRID_RES + cz * GRID_RES * GRID_RES;
	}

	// STEP 4: Sort the triangle indices by cell code.
	std::sort(indices.begin(), indices.end(), [&](int a, int b) {
		return cellCodes[a] < cellCodes[b];
		});

	// STEP 5: Define a recursive lambda that builds a BVH node over the range [start, end] of the sorted indices.
	// We also use SAH to choose the best split point and spawn parallel tasks if the workload is large.
	std::function<BVHNode* (int, int)> buildRec = [&](int start, int end) -> BVHNode* {
		BVHNode* node = new BVHNode();
		node->bounds.reset();
		for (int i = start; i <= end; i++) {
			int tid = indices[i];
			node->bounds.extend(inputTriangles[tid].vertices[0].p);
			node->bounds.extend(inputTriangles[tid].vertices[1].p);
			node->bounds.extend(inputTriangles[tid].vertices[2].p);
		}
		int count = end - start + 1;
		if (count <= MAXNODE_TRIANGLES) {
			// Leaf node: store the indices.
			node->triangleIndices.assign(indices.begin() + start, indices.begin() + end + 1);
			return node;
		}

		// STEP 6: Precompute prefix and suffix AABBs for SAH cost evaluation.
		std::vector<AABB> prefix(count);
		std::vector<AABB> suffix(count);
		prefix[0] = AABB(inputTriangles[indices[start]]);
		for (int i = 1; i < count; i++) {
			int tid = indices[start + i];
			prefix[i] = prefix[i - 1];
			prefix[i].extend(inputTriangles[tid].vertices[0].p);
			prefix[i].extend(inputTriangles[tid].vertices[1].p);
			prefix[i].extend(inputTriangles[tid].vertices[2].p);
		}
		suffix[count - 1] = AABB(inputTriangles[indices[end]]);
		for (int i = count - 2; i >= 0; i--) {
			int tid = indices[start + i];
			suffix[i] = suffix[i + 1];
			suffix[i].extend(inputTriangles[tid].vertices[0].p);
			suffix[i].extend(inputTriangles[tid].vertices[1].p);
			suffix[i].extend(inputTriangles[tid].vertices[2].p);
		}

		// STEP 7: Choose the best split based on SAH.
		float bestCost = FLT_MAX;
		int bestSplit = start; // index in the sorted array
		for (int i = 0; i < count - 1; i++) {
			float leftArea = prefix[i].area();
			float rightArea = suffix[i + 1].area();
			float cost = leftArea * (i + 1) + rightArea * (count - i - 1);
			if (cost < bestCost) {
				bestCost = cost;
				bestSplit = start + i;
			}
		}

		// STEP 8: Recurse in parallel if the workload is high.
		BVHNode* leftChild;
		BVHNode* rightChild;
		if (count > PARALLEL_THRESHOLD) {
			auto futureLeft = std::async(std::launch::async, buildRec, start, bestSplit);
			rightChild = buildRec(bestSplit + 1, end);
			leftChild = futureLeft.get();
		}
		else {
			leftChild = buildRec(start, bestSplit);
			rightChild = buildRec(bestSplit + 1, end);
		}
		node->l = leftChild;
		node->r = rightChild;
		return node;
		};

	// STEP 9: Build the tree recursively using the lambda.
	BVHNode* tempRoot = buildRec(0, nTriangles - 1);
	// Move the temporary root's data into "this" node.
	*this = *tempRoot;
	delete tempRoot;
}

*/