//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Nagy Zalán
// Neptun : V9T3UL
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
//=============================================================================================
// Computer Graphics Sample Program: Ray-tracing-let
//=============================================================================================
#include "framework.h"

/*
	TODO
	Kupbol jovo feny és a kup belseje
	attehetoseg
*/

struct Material {
	vec3 ka, kd, ks;
	float  shininess;
	Material(vec3 _kd, vec3 _ks, float _shininess) : ka(_kd* M_PI), kd(_kd), ks(_ks) { shininess = _shininess; }
};

struct Hit {
	float t;
	vec3 position, normal;
	Material* material;
	Hit() { t = -1; }
};

struct Ray {
	vec3 start, dir;
	Ray(vec3 _start, vec3 _dir) { start = _start; dir = normalize(_dir); }
};

class Intersectable {
protected:
	Material* material;
public:
	virtual Hit intersect(const Ray& ray) = 0;
};


struct Sqare : public Intersectable {
	vec3 a, b, c, d;
	Sqare(const vec3& _r1, const vec3& _r2, const vec3& _r3, const vec3& _r4) {
		a = _r1;
		b = _r2;
		c = _r3;
		d = _r4;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 n = cross((b - a), (c - a));
		float t = dot((a - ray.start), n) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;

		if (dot(cross(b - a, p - a), n) > 0
			&& dot(cross(c - b, p - b), n) > 0
			&& dot(cross(d - c, p - c), n) > 0
			&& dot(cross(a - d, p - d), n) > 0
			)
		{
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normalize(n);
		}
		return hit;
	}
};

struct Triangle : public Intersectable {
	vec3 r1, r2, r3;
	Triangle(const vec3& _r1, const vec3& _r2, const vec3& _r3) {
		r1 = _r1;
		r2 = _r2;
		r3 = _r3;
	}
	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 n = cross((r2 - r1), (r3 - r1));
		float t = dot((r1 - ray.start), n) / dot(ray.dir, n);
		if (t < 0) return hit;
		vec3 p = ray.start + ray.dir * t;

		if (dot(cross(r2 - r1, p - r1), n) > 0 && dot(cross(r3 - r2, p - r2), n) > 0 && dot(cross(r1 - r3, p - r3), n) > 0) {
			hit.t = t;
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normalize(n);
		}
		return hit;
	}
};

struct Cone : public Intersectable {
	vec3 p;
	vec3 n;
	Cone(vec3 _p, vec3 _n) {
		p = _p;
		n = _n;
	}
	
	Hit intersect(const Ray& ray) {
		Hit hit;

		vec3 d = ray.dir;
		vec3 s = ray.start;
		float a = pow(dot(d,n),2)- dot(d, d) * pow(cos(0.3),2);
		float b = 2 * dot(d, n) * dot(n, s) - 2 * dot(d, n) * dot(n, p) -2* pow(cos(0.3), 2)*( dot(d, s) - dot(d, p));
		float c = pow(dot(s, n), 2) + pow(dot(p, n), 2) - 2 * (dot(s, p) * dot(n, n)) - pow(cos(0.3), 2) * (dot(s, s) + dot(p, p) - 2 * dot(s, p));
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0 && t2<=0) return hit;

		float kisebbt, nagyobbt;
		if (t1 < t2) {
			kisebbt = t1;
			nagyobbt = t2;
		}
		else {
			kisebbt = t2;
			nagyobbt = t1;
		}

		if (dot(((ray.start + ray.dir * nagyobbt) - p), n) > 0 && dot(((ray.start + ray.dir * nagyobbt) - p), n) <0.4) {
			hit.t = nagyobbt;
			if (dot(((ray.start + ray.dir * kisebbt) - p), n) > 0 && dot(((ray.start + ray.dir * kisebbt) - p), n) < 0.4) {
				hit.t = kisebbt;
			}
			hit.position = ray.start + ray.dir * hit.t;
			hit.normal = normalize(cross(cross(n, hit.position), hit.position));
		}

		return hit;
	}
};

struct Sphere : public Intersectable {
	vec3 center;
	float radius;

	Sphere(const vec3& _center, float _radius, Material* _material) {
		center = _center;
		radius = _radius;
		material = _material;
	}

	Hit intersect(const Ray& ray) {
		Hit hit;
		vec3 dist = ray.start - center;
		float a = dot(ray.dir, ray.dir);
		float b = dot(dist, ray.dir) * 2.0f;
		float c = dot(dist, dist) - radius * radius;
		float discr = b * b - 4.0f * a * c;
		if (discr < 0) return hit;
		float sqrt_discr = sqrtf(discr);
		float t1 = (-b + sqrt_discr) / 2.0f / a;	// t1 >= t2 for sure
		float t2 = (-b - sqrt_discr) / 2.0f / a;
		if (t1 <= 0) return hit;
		hit.t = (t2 > 0) ? t2 : t1;
		hit.position = ray.start + ray.dir * hit.t;
		hit.normal = (hit.position - center) * (1.0f / radius);
		hit.material = material;
		return hit;
	}
};


class Camera {
	vec3 eye, lookat, right, up;
	float fov;
public:
	void set(vec3 _eye, vec3 _lookat, vec3 vup, float _fov) {
		eye = _eye;
		lookat = _lookat;
		fov = _fov;
		vec3 w = eye - lookat;
		float focus = length(w);
		right = normalize(cross(vup, w)) * focus * tanf(fov / 2);
		up = normalize(cross(w, right)) * focus * tanf(fov / 2);
	}
	Ray getRay(int X, int Y) {
		vec3 dir = lookat + right * (2.0f * (X + 0.5f) / windowWidth - 1) + up * (2.0f * (Y + 0.5f) / windowHeight - 1) - eye;
		return Ray(eye, dir);
	}
	vec3 getEye() {
		return eye;
	}
};

struct Light {
	vec3 direction, position;
	vec3 Le;
	Light(vec3 _direction,vec3 _position , vec3 _Le) {
		direction = normalize(_direction);
		position = _position;
		Le = _Le;
	}
};

float rnd() { return (float)rand() / RAND_MAX; }

const float epsilon = 0.0001f;

class Scene {
	std::vector<Intersectable*> objects;
	std::vector<Light*> lights;
	Camera camera;
	vec3 La = vec3(0, 0, 0);
public:
	void build() {
		vec3 eye = vec3(-2, 0, 3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		//vec3 eye = vec3(0, 0, 3), vup = vec3(0, 1, 0), lookat = vec3(0, 0, 0);
		float fov = 45 * M_PI / 180;
		camera.set(eye, lookat, vup, fov);

		La = vec3(0.2f, 0.2f, 0.2f);
		vec3 lightDirection(-1, 0, 0), Le(0.5, 0, 0);
		lights.push_back(new Light(lightDirection, vec3(0,0.8,0), Le));

		//vec3 lightDirection1(-1, 0, 0), Le1(0, 0.5, 0);
		//lights.push_back(new Light(lightDirection1, vec3(0, 0, 0), Le1));

		
		vec3 kd(0.3f, 0.2f, 0.1f), ks(2, 2, 2);
		Material* material = new Material(kd, ks, 50);

		//szoba sarkai
		vec3 a = vec3(-1, -1, 1);
		vec3 b = vec3(1, -1, 1);
		vec3 c = vec3(1, 1, 1);
		vec3 d = vec3(-1, 1, 1);
		vec3 e = vec3(-1, -1, -1);
		vec3 f = vec3(1, -1, -1);
		vec3 g = vec3(1, 1, -1);
		vec3 h = vec3(-1, 1, -1);
		//négy fala a szobának, ami látszik
		objects.push_back(new Sqare(e, f, g, h));
		objects.push_back(new Sqare(b, f, g, c));
		objects.push_back(new Sqare(d, c, g, h));
		objects.push_back(new Sqare(a, b, f, e));

		objects.push_back(new Sqare(a, d, h, e));
		objects.push_back(new Sqare(a, b, c, d));



		vec3 icosahedron[12] = {
vec3(0+1, -0.525731-1,  0.850651+1),
vec3(0.850651+1,  0-1,  0.525731+1),
vec3(0.850651+1,  0-1, -0.525731+1),
vec3(-0.850651+1,  0-1, -0.525731+1),
vec3(-0.850651+1,  0-1,  0.525731+1),
vec3(-0.525731+1,  0.850651-1,  0+1),
vec3(0.525731+1,  0.850651-1,  0+1),
vec3(0.525731+1, -0.850651-1,  0+1),
vec3(-0.525731+1, -0.850651-1,  0+1),
vec3(0+1, -0.525731-1, -0.850651+1),
vec3(0+1,  0.525731-1, -0.850651+1),
vec3(0+1,  0.525731-1,  0.850651+1)
	};

		int icosahedron_order[60] = {
	2,  3,  7,
	2,  8,  3,
	4,  5,  6,
	5,  4,  9,
	7,  6,  12,
	6,  7,  11,
	10,  11,  3,
	11,  10,  4,
	8,  9,  10,
	9,  8,  1,
	12,  1,  2,
	1,  12,  5,
	7,  3,  11,
	2,  7,  12,
	4,  6,  11,
	6,  5,  12,
	3,  8,  10,
	8,  2,  1,
	4,  10,  9,
	5,  9,  1
	};

		vec3 dodecahedron[20] = {
	vec3(-0.57735-0.5, -0.57735-1, 0.57735),
	vec3(0.934172- 0.5, 0.356822-1, 0),
	vec3(0.934172- 0.5, -0.356822-1, 0),
	vec3(-0.934172- 0.5, 0.356822-1, 0),
	vec3(-0.934172- 0.5, -0.356822-1, 0),
	vec3(0- 0.5, 0.934172-1, 0.356822),
	vec3(0- 0.5, 0.934172-1, -0.356822),
	vec3(0.356822- 0.5, 0-1, -0.934172),
	vec3(-0.356822- 0.5, 0-1, -0.934172),
	vec3(0- 0.5, -0.934172-1, -0.356822),
	vec3(0- 0.5, -0.934172-1, 0.356822),
	vec3(0.356822- 0.5, 0-1, 0.934172),
	vec3(-0.356822- 0.5, 0-1, 0.934172),
	vec3(0.57735- 0.5, 0.57735-1, -0.57735),
	vec3(0.57735- 0.5, 0.57735-1, 0.57735),
	vec3(-0.57735- 0.5, 0.57735-1, -0.57735),
	vec3(-0.57735-0.5, 0.57735-1, 0.57735),
	vec3(0.57735- 0.5, -0.57735-1, -0.57735),
	vec3(0.57735- 0.5, -0.57735-1, 0.57735),
	vec3(-0.57735- 0.5, -0.57735-1, -0.57735)
	};

		int dodecahedron_order[108] = {
		19, 3, 2,
		12, 19, 2,
		15, 12, 2,
		8, 14, 2,
		18, 8, 2,
		3, 18, 2,
		20, 5, 4,
		9, 20, 4,
		16, 9, 4,
		13, 17, 4,
		1, 13, 4,
		5, 1, 4,
		7, 16, 4,
		6, 7, 4,
		17, 6, 4,
		6, 15, 2,
		7, 6, 2,
		14, 7, 2,
		10, 18, 3,
		11, 10, 3,
		19, 11, 3,
		11, 1, 5,
		10, 11, 5,
		20, 10, 5,
		20, 9, 8,
		10, 20, 8,
		18, 10, 8,
		9, 16, 7,
		8, 9, 7,
		14, 8, 7,
		12, 15, 6,
		13, 12, 6,
		17, 13, 6,
		13, 1, 11,
		12, 13, 11,
		19, 12, 11
	};

		//icosahedron
		for (int i = 0; i < 20; i++) {
			objects.push_back(new Triangle(icosahedron[icosahedron_order[3*i]-1]*0.5, icosahedron[icosahedron_order[3*i+1]-1]*0.5, icosahedron[icosahedron_order[3*i+2]-1]*0.5));
		}

		//dodecahedron
		for (int i = 0; i < 36; i++) {
			objects.push_back(new Triangle(dodecahedron[dodecahedron_order[3 * i] - 1] * 0.5, dodecahedron[dodecahedron_order[3 * i + 1] - 1] * 0.5, dodecahedron[dodecahedron_order[3 * i + 2] - 1] * 0.5));
		}


		objects.push_back(new Cone(vec3(0, 1.0f, 0.0f), vec3(0, -1, 0)));

		objects.push_back(new Sphere(vec3(0, 0, 0.8f), 0.2f, material));

	}

	void render(std::vector<vec4>& image) {
		for (int Y = 0; Y < windowHeight; Y++) {
#pragma omp parallel for
			for (int X = 0; X < windowWidth; X++) {
				vec3 color = trace(camera.getRay(X, Y));
				image[Y * windowWidth + X] = vec4(color.x, color.y, color.z, 1);
			}
		}
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (bestHit.t < 0 || hit.t < bestHit.t))  bestHit = hit;
		}
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}

	Hit secondIntersect(Ray ray) {
		Hit bestHit= firstIntersect(ray);

		Hit worstHit;
		for (Intersectable* object : objects) {
			Hit hit = object->intersect(ray); //  hit.t < 0 if no intersection
			if (hit.t > 0 && (worstHit.t < 0 || hit.t < worstHit.t) && hit.t!=bestHit.t)  worstHit = hit;
		}

		if (dot(ray.dir, worstHit.normal) > 0) worstHit.normal = worstHit.normal * (-1);
		return worstHit;
	}


	bool shadowIntersect(Ray ray) {	// for directional lights
		for (Intersectable* object : objects) if (object->intersect(ray).t > 0) return true;
		return false;
	}

	vec3 trace(Ray ray, int depth = 0) {
		Hit hit = secondIntersect(ray);
		if (hit.t < 0) return La;

		float L = 0.2 * (1 + dot(-1*normalize(hit.normal), normalize(ray.dir)));
		vec3 outRadiance =  vec3(L,L,L);

		for (Light* light : lights) {
			Ray shadowRay(hit.position + hit.normal * epsilon, light->direction);
			float cosTheta = dot(hit.normal, light->direction );
			if (cosTheta > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
				outRadiance = outRadiance + light->Le * vec3(L, L, L) * cosTheta;
				vec3 halfway = normalize(-ray.dir + light->direction );
				float cosDelta = dot(hit.normal, halfway);
				if (cosDelta > 0) outRadiance = outRadiance + light->Le * vec3(L, L, L);
			}
		}


		Ray feny(vec3(0, 0, 0), vec3(0,1,0));
		float cos = dot(hit.normal, feny.dir);
		Ray shadowRay(hit.position + hit.normal * epsilon, feny.dir);
		if (cos > 0 && !shadowIntersect(shadowRay)) {	// shadow computation
			outRadiance = outRadiance + vec3(0, 1, 0) * vec3(L, L, L);
		}

		return outRadiance;
	}
};

GPUProgram gpuProgram; // vertex and fragment shaders
Scene scene;

// vertex shader in GLSL
const char* vertexSource = R"(
	#version 330
    precision highp float;

	layout(location = 0) in vec2 cVertexPosition;	// Attrib Array 0
	out vec2 texcoord;

	void main() {
		texcoord = (cVertexPosition + vec2(1, 1))/2;							// -1,1 to 0,1
		gl_Position = vec4(cVertexPosition.x, cVertexPosition.y, 0, 1); 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char* fragmentSource = R"(
	#version 330
    precision highp float;

	uniform sampler2D textureUnit;
	in  vec2 texcoord;			// interpolated texture coordinates
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = texture(textureUnit, texcoord); 
	}
)";

class FullScreenTexturedQuad {
	unsigned int vao;	// vertex array object id and texture id
	Texture texture;
public:
	FullScreenTexturedQuad(int windowWidth, int windowHeight, std::vector<vec4>& image)
		: texture(windowWidth, windowHeight, image)
	{
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo;		// vertex buffer objects
		glGenBuffers(1, &vbo);	// Generate 1 vertex buffer objects

		// vertex coordinates: vbo0 -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo); // make it active, it is an array
		float vertexCoords[] = { -1, -1,  1, -1,  1, 1,  -1, 1 };	// two triangles forming a quad
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified 
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);     // stride and offset: it is tightly packed
	}

	void Draw() {
		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		gpuProgram.setUniform(texture, "textureUnit");
		glDrawArrays(GL_TRIANGLE_FAN, 0, 4);	// draw two triangles forming a quad
	}
};

FullScreenTexturedQuad* fullScreenTexturedQuad;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	scene.build();

	std::vector<vec4> image(windowWidth * windowHeight);
	scene.render(image);

	// copy image to GPU as a texture
	fullScreenTexturedQuad = new FullScreenTexturedQuad(windowWidth, windowHeight, image);

	// create program for the GPU
	gpuProgram.create(vertexSource, fragmentSource, "fragmentColor");
}

// Window has become invalid: Redraw
void onDisplay() {
	fullScreenTexturedQuad->Draw();
	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
}



