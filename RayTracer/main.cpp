#include "rtweekend.h"

#include "camera.h"
#include "color.h"
#include "hittable_list.h"
#include "material.h"
#include "sphere.h"

#include <iostream>
#include <fstream>
#include <thread>
#include <vector>
#include <time.h>

#define MIN_HIT_LIMIT 0.001
#define THREAD_COUNT 20
#define IMAGE_WIDTH 300
#define ASPECT_RATIO 1.5 // 3:2
#define SAMPLES_PER_PIXEL 10
#define MAX_DEPTH 30

int current_progress = 0;
int total_progress = 100;

color ray_color(const ray& r, const hittable& world, int depth) {
	hit_record rec;

	// if we've exceeded the ray bounce limit, no more light is gathered.  
	if (depth <= 0)
		return color(0, 0, 0);

	if (world.hit(r, MIN_HIT_LIMIT, infinity, rec)) {
		ray scattered;
		color attenuation;
		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * ray_color(scattered, world, depth - 1);
		return color(0, 0, 0);
	}

	vec3 unit_direction = unit_vector(r.direction());
	auto t = 0.5*(unit_direction.y() + 1.0);
	return (1.0 - t)*color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
}

hittable_list random_scene() {
	hittable_list world;

	auto ground_material = make_shared<lambertian>(color(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(point3(0, -1000, 0), 1000, ground_material));

	for (int a = -3; a < 3; a++) {
		for (int b = -3; b < 3; b++) {
			auto choose_mat = random_double();
			point3 center(a + 0.9*random_double(), 0.2, b + 0.9*random_double());

			if ((center - point3(4, 0.2, 0)).length() > 0.9) {
				shared_ptr<material> sphere_material;

				if (choose_mat < 0.8) {
					// diffuse
					auto albedo = color::random() * color::random();
					sphere_material = make_shared<lambertian>(albedo);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
				else if (choose_mat < 0.95) {
					// metal
					auto albedo = color::random(0.5, 1);
					auto fuzz = random_double(0, 0.5);
					sphere_material = make_shared<metal>(albedo, fuzz);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
				else {
					// glass
					sphere_material = make_shared<dielectric>(1.5);
					world.add(make_shared<sphere>(center, 0.2, sphere_material));
				}
			}
		}
	}

	auto material1 = make_shared<dielectric>(1.5);
	world.add(make_shared<sphere>(point3(0, 1, 0), 1.0, material1));

	auto material2 = make_shared<lambertian>(color(0.4, 0.2, 0.1));
	world.add(make_shared<sphere>(point3(-4, 1, 0), 1.0, material2));

	auto material3 = make_shared<metal>(color(0.7, 0.6, 0.5), 0.0);
	world.add(make_shared<sphere>(point3(4, 1, 0), 1.0, material3));

	return world;
}

color** image_matrix(const int width, const int height) {
	color** arr = (color**) calloc(height, sizeof(color*));
	for (int j = height-1; j >= 0; j--) {
		arr[j] = (color*)calloc(width, sizeof(color));
		for (int i = 0; i < width; i++)
			arr[j][i] = color();
	}
	return arr;
}

void update_progress() {
	current_progress++;
	std::cerr << "\rProcessing row: " << current_progress << '/' << total_progress << ' ' << std::flush;
}

void process_matrix_rows(const int start_row, const int end_row, const int image_width, const int image_height, color** output_matrix, const int samples_per_pixel, const int max_depth, const hittable_list& world, const camera& cam) {

	for (int j = start_row; j <= end_row; j++) {
		update_progress();

		for (int i = 0; i < image_width; i++) {
			color pixel_color(0, 0, 0);

			for (int s = 0; s < samples_per_pixel; s++) {
				auto u = (i + random_double()) / (image_width - 1);
				auto v = (j + random_double()) / (image_height - 1);
				ray r = cam.get_ray(u, v);
				pixel_color += ray_color(r, world, max_depth);
			}
			output_matrix[j][i] = format_color(pixel_color, samples_per_pixel);
		}
	}
}

void multithreaded_raytracing(const int thread_count, const int image_width, const int image_height, color** output_matrix, const int samples_per_pixel, const int max_depth, const hittable_list& world, const camera& cam) {
	std::vector<std::thread> thread_list;

	int cur_row = 0;
	int row_segment = image_height / thread_count;
	for (int tc = thread_count-1; tc >= 0; tc--) {
		thread_list.push_back(std::thread(process_matrix_rows, cur_row, cur_row + row_segment - 1, image_width, image_height, output_matrix, samples_per_pixel, max_depth, world, cam));
		cur_row += row_segment;
	}

	for (int tc = thread_count - 1; tc >= 0; tc--)
		thread_list[tc].join();
}

int main() {
	// image
	const auto aspect_ratio = ASPECT_RATIO;
	const int image_width = IMAGE_WIDTH;
	const int image_height = static_cast<int>(image_width / aspect_ratio);
	const int samples_per_pixel = SAMPLES_PER_PIXEL;
	const int max_depth = MAX_DEPTH;

	// world
	auto world = random_scene();

	// camera
	point3 lookfrom(13, 2, 3);
	point3 lookat(0, 0, 0);
	vec3 vup(0, 1, 0);
	auto dist_to_focus = 10.0;
	auto aperture = 0.1;

	camera cam(lookfrom, lookat, vup, 20, aspect_ratio, aperture, dist_to_focus);

	// initialise progress
	current_progress = 0;
	total_progress = image_height;

	// render
	color** output_matrix = image_matrix(image_width, image_height);
	clock_t tStart = clock();
	multithreaded_raytracing(THREAD_COUNT, image_width, image_height, output_matrix, samples_per_pixel, max_depth, world, cam);
	clock_t tEnd = clock();
	
	// setup output file stream
	std::ofstream outdata;
	outdata.open("image.ppm");
	if (!outdata) {
		// file could not be opened
		std::cerr << "Error: output file could not be opened.\n";
		exit(1);
	}

	// dump output
	outdata << "P3\n" << image_width << ' ' << image_height << "\n255\n";

	for (int j = image_height-1; j >= 0; --j) {
		std::cerr << "\rStoring row: " << j << ' ' << std::flush;
		for (int i = 0; i < image_width; i++) {
			outdata << static_cast<int>(output_matrix[j][i].x()) << ' ' << static_cast<int>(output_matrix[j][i].y()) << ' ' << static_cast<int>(output_matrix[j][i].z()) << '\n';
		}
	}

	// close output stream
	outdata.close();

	std::cerr << "\nDone.\n";
	std::cout << "\nProcessing Time: " << (double)(tEnd - tStart) / CLOCKS_PER_SEC << "s\n";
	std::cout << "Ray Tracing completed.\n";

	return 0;
}