#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>

#include "frac_cpu.h"

void execute_and_print_summary (const FractalZooming& zooming) {}

template<typename Frac, typename... TNext>
void execute_and_print_summary (FractalZooming& zooming, Frac& frac, TNext&... next) {
	std::cout << "Executing '" << frac.name () << "' ..." << std::endl;
	frac.execute (zooming);
	std::cout << "Done!" << std::endl;

	auto timer = frac.timer ();
	std::cout << frac.name () << " (total): " << timer.total ().count () << "s" << std::endl;
	for (const auto& e : timer.times ()) {
		std::cout
			<< " - '" << std::get<0> (e) << "' "
			<< std::get<1> (e).count ()
			<< "s" << std::endl;
	}
	std::cout << "\n";

	execute_and_print_summary (zooming, next...);
}

template<typename FracA>
void compare_times_internal (const FracA& base) { }

template<typename FracA, typename FracB, typename... TNext>
void compare_times_internal (const FracA& base, const FracB& other, const TNext&... others) {
	std::cout << " - " << other.name () << ": "
		<< std::setprecision (2) << std::fixed
		<< base.timer ().total_in_ms () / other.timer ().total_in_ms ()
		<< std::endl;

	compare_times_internal (base, others...);
}

template<typename FracA, typename FracB, typename... TNext>
void compare_times (const FracA& base, const FracB& other, const TNext&... others) {
	std::cout << "Speedup (compared to " << base.name () << ":" << std::endl;
	compare_times_internal (base, other, others...);
}

template<typename Frac, size_t max_value = 64>
std::tuple<size_t, std::chrono::duration<double>> find_optimal (FractalZooming& zooming, std::function<Frac (size_t)> creator) {
	std::cout << "Finding best ...";
	size_t best = 0;
	std::chrono::duration<double> best_time = std::chrono::duration<double>::max ();
	for (size_t i = 1; i <= max_value; i *= 2)
	{
		auto frac = creator (i);
		frac.execute (zooming);

		auto time = frac.timer ().total ();
		if (time < best_time) {
			best = i;
			best_time = time;
		}
	}
	std::cout << " Done!" << std::endl;

	return std::make_tuple (best, best_time);
}

template<typename Frac, size_t max_value = 64>
std::tuple<size_t, size_t, std::chrono::duration<double>> find_optimal (FractalZooming& zooming, std::function<Frac (size_t, size_t)> creator) {
	std::cout << "Finding best ...";
	std::tuple<size_t, size_t> best;
	std::chrono::duration<double> best_time = std::chrono::duration<double>::max ();
	for (size_t i = 1; i <= max_value; i *= 2)
	{
		for (size_t j = 1; j <= max_value; j *= 2)
		{
			auto frac = creator (i, j);
			frac.execute (zooming);

			auto time = frac.timer ().total ();
			if (time < best_time) {
				best = std::make_tuple (i, j);
				best_time = time;
			}
		}
	}
	std::cout << " Done!" << std::endl;

	return std::make_tuple (std::get<0> (best), std::get<1> (best), best_time);
}

FractalZooming create_zooming () {
	std::cout << "Generating color map ...";
	pixel_t inside_col{ 255, 255, 255, 0 };
	pixel_t outside_col{ 0, 0, 0, 0 };

	FractalZooming fractal_zoom{
		complex_t{ -2.74529004f, -1.01192498f },
		complex_t{ 1.25470996f, 1.23807502f },
		0.95f,
		200,
		complex_t{ -0.745289981f, 0.113075003f },
		FractalZooming::SaveImage::No
	};
	for (size_t i = 0; i < COLOR_COUNT; i++)
	{
		fractal_zoom.color_map[i] = interpolate (outside_col, inside_col, i * 1.0 / COLOR_COUNT);
	}
	std::cout << " Done!" << std::endl;
	return fractal_zoom;
}

void find_best () {
	int image_width = 1024; int image_height = 576;

	std::cout << "Running               : find_best" << std::endl;
	std::cout << "Resolution            : " << image_width << " x " << image_height << " pixels\n" << std::endl;
	
	auto fractal_zoom = create_zooming ();
	std::cout << std::endl;

	auto best_glsp = find_optimal<FracCPU_GSLP<FracUseCPUExt::AVX_FMA, 4, FracProgress::None>> (fractal_zoom, [image_width, image_height](size_t t) {
		return FracCPU_GSLP<FracUseCPUExt::AVX_FMA, 4, FracProgress::None> (image_width, image_height, t);
		});
	auto best_gpls = find_optimal<FracCPU_GPLS<FracUseCPUExt::AVX_FMA, 4, FracProgress::None>> (fractal_zoom, [image_width, image_height](size_t t) {
		return FracCPU_GPLS<FracUseCPUExt::AVX_FMA, 4, FracProgress::None> (image_width, image_height, t);
		});
	auto best_gplp = find_optimal<FracCPU_GPLP<FracUseCPUExt::AVX_FMA, 4, FracProgress::None>> (fractal_zoom, [image_width, image_height](size_t i, size_t t) {
		return FracCPU_GPLP<FracUseCPUExt::AVX_FMA, 4, FracProgress::None> (image_width, image_height, i, t);
		});
	std::cout << "Best:\n"
		<< "GSLP: tasks = " << std::get<0> (best_glsp) << " with " << std::get<1> (best_glsp).count () << "s\n"
		<< "GPLS: tasks = " << std::get<0> (best_gpls) << " with " << std::get<1> (best_gpls).count () << "s\n"
		<< "GPLP: images = " << std::get<0> (best_gplp) << ", tasks = " << std::get<1> (best_gplp) << " with " << std::get<2> (best_gplp).count () << "s\n"
		;
	/*
	Best: (1024 x 576 pixels)
		GSLP: tasks = 32 with 1.41572s
		GPLS: tasks = 64 with 1.43518s
		GPLP: images = 64, tasks = 16 with 1.33578s
	Best: (2048 x 1152 pixels)
		GSLP: tasks = 32 with 5.46838s
		GPLS: tasks = 32 with 5.49379s
		GPLP: images = 32, tasks = 32 with 5.19711s
	*/
}

void test_bed () {
	// target res: 8.192 x 4.608
	//int image_width = 8192; int image_height = 4608;
	//int image_width = 4096; int image_height = 2304;
	//int image_width = 2048; int image_height = 1152;
	int image_width = 1024; int image_height = 576;

	std::cout << "Running               : test_bed" << std::endl;
	std::cout << "Resolution            : " << image_width << " x " << image_height << " pixels\n" << std::endl;

	auto fractal_zoom = create_zooming ();
	fractal_zoom.save_images = FractalZooming::SaveImage::ToDisk;
	std::cout << std::endl;

	// Current best: GPLP
	// On FH: GPLS
	FracCPU_GPLP<FracUseCPUExt::AVX_FMA, 8> frac_cpu{ image_width, image_height,
		64, std::thread::hardware_concurrency() };
	FracCPU_GPLS<FracUseCPUExt::AVX_FMA, 8> frac_cpu_gpls{ image_width, image_height,
		64};
	FracCPU_GSLP<FracUseCPUExt::AVX_FMA, 8> frac_cpu_gslp{ image_width, image_height, 64};

	execute_and_print_summary (fractal_zoom, frac_cpu_gslp);
	return;

	// execute_and_print_summary (fractal_zoom,
	// 	frac_cpu,
	// 	frac_cpu_gpls,
	// 	frac_cpu_gslp,
	// 	frac_gpu);
	// compare_times (frac_cpu, frac_cpu_gpls, frac_cpu_gslp, frac_gpu);
}

int main () {
	print_cpu_summary ();

	// std::cout << "\nSetting process priority to High ...";
	// SetPriorityClass (GetCurrentProcess (), HIGH_PRIORITY_CLASS);
	// std::cout << " Done!\n" << std::endl;

	test_bed ();
}
