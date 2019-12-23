#pragma once

#include <vector>
#include <array>
#include <tuple>
#include <map>
#include <thread>
#include <future>

#include "animated_gif.h"
#include "frac.h"
#include "timer.h"
#include "parallelizer.h"

#include "instruction_set.h"
#include "immintrin.h"

using namespace std::string_view_literals;
using namespace std::literals::chrono_literals;

const InstructionSet::InstructionSet_Internal InstructionSet::CPU_Rep;
void print_cpu_summary () {
	std::cout << std::boolalpha;
	std::cout << "CPU: " << std::endl;
	std::cout << "  * CPU Vendor        : " << InstructionSet::Vendor () << std::endl;
	std::cout << "  * CPU Brand         : " << InstructionSet::Brand () << std::endl;
	std::cout << "  * Threads           : " << std::thread::hardware_concurrency () << std::endl;
	std::cout << "  * Supports AVX?     : " << InstructionSet::AVX () << std::endl;
	std::cout << "  * Supports AVX2?    : " << InstructionSet::AVX2 () << std::endl;
	std::cout << "  * Supports FMA?     : " << InstructionSet::FMA () << std::endl;
}

enum class FracUseCPUExt {
	None,
	AVX,
	AVX_FMA
};

enum class FracProgress {
	None,
	Cout
};



/**
Fractal Zoom implementation for CPU.
Can use AVX and FMA extensions
*/
template<
	FracUseCPUExt cpu_ext = FracUseCPUExt::AVX_FMA,
	/** Defines how many pixels (multiplied by 8) are filled in a row */
	int pixels_size = 1,
	FracProgress report_progress = FracProgress::Cout
>
class FracCPU {
protected:
	std::string _name;
	int _image_width;
	int _image_height;
	Timer _timer;

public:
	FracCPU (int image_width, int image_height)
		: FracCPU (image_width, image_height, "FracCPU (1 thread)") { }

protected:
	FracCPU (int image_width, int image_height, std::string name)
		: _image_width{ image_width }, _image_height{ image_height }, _name{ name } {
		switch (cpu_ext)
		{
			case FracUseCPUExt::AVX:
				_name += "+AVX";
				break;
			case FracUseCPUExt::AVX_FMA:
				_name += "+AVX+FMA";
				break;
		}
		_name += " (" + std::to_string (8 * pixels_size) + " pixels)";
	}

public:
	void execute (const FractalZooming& zooming) {
		std::vector<pixel_t> image(_image_width * _image_height);

		_timer.start ("all");

		auto lower_left{ zooming.start_lower_left };
		auto upper_right{ zooming.start_upper_right };

		for (size_t i = 0; i < zooming.zoom_steps; i++)
		{
			auto scale = compute_scale (lower_left, upper_right, image);

			// image starts at lower left conrer
			for (size_t y = 0; y < image.height (); y++)
			{
				fill_row (y, image, zooming, lower_left, scale);
			}

			if (zooming.save_images == FractalZooming::SaveImage::ToDisk) {
				std::string file_name;
				file_name += "frac_zoom_";
				file_name += std::to_string (i);
				file_name += ".bmp";

				image.to_file (file_name);
			}

			zoom_and_re_center_inplace (lower_left, upper_right, zooming);
		}

		_timer.stop ();
	}

	const std::string& name () const {
		return _name;
	}

	const Timer& timer () const {
		return _timer;
	}

protected:
	std::map<size_t, std::tuple<complex_t, complex_t>> get_bounds (const FractalZooming& zooming) {
		std::map<size_t, std::tuple<complex_t, complex_t>> bounds;
		bounds[0] = std::make_tuple (zooming.start_lower_left, zooming.start_upper_right);
		for (size_t i = 1; i < zooming.zoom_steps; i++)
		{
			bounds[i] = zoom_and_re_center (
				std::get<0> (bounds[i - 1]),
				std::get<1> (bounds[i - 1]),
				zooming
			);
		}
		return bounds;
	}

	inline 
	void fill_row (size_t y, std::vector<pixel_t>& image, const FractalZooming& zooming, const complex_t& lower_left, const std::array<float, 2>& scale) {
		if constexpr (cpu_ext == FracUseCPUExt::AVX || cpu_ext == FracUseCPUExt::AVX_FMA) {
			for (size_t x = 0; x < _image_width; x += 8 * pixels_size)
			{
				if constexpr (pixels_size == 1)
					fill_8_pixels (x, y, image, zooming, lower_left, scale);
				else
					fill_pixels<pixels_size> (x, y, image, zooming, lower_left, scale);
			}
		}
		else {
			for (size_t x = 0; x < _image_width; x++)
			{
				auto idx = y * _image_width + x;
				auto c = idx_to_complex (x, y, lower_left, scale);
				auto result = mandelbrot (c);

				image[idx] = get_color (result, zooming);
			}
		}
	}

	inline 
	void fill_8_pixels (size_t x, size_t y, std::vector<pixel_t>& image, const FractalZooming& zooming, const complex_t& lower_left, const std::array<float, 2>& scale) {
		auto base_idx = y * _image_width + x;

		__m256 c_real;
		__m256 c_imag;
		if constexpr (cpu_ext == FracUseCPUExt::AVX_FMA) {
			auto c = idx_to_complex_8 (x, y, lower_left, scale);
			c_real = std::get<0> (c);
			c_imag = std::get<1> (c);
		}
		else {
			auto c = std::array<complex_t, 8>{
				idx_to_complex (x, y, lower_left, scale),
				idx_to_complex (x + 1, y, lower_left, scale),
				idx_to_complex (x + 2, y, lower_left, scale),
				idx_to_complex (x + 3, y, lower_left, scale),
				idx_to_complex (x + 4, y, lower_left, scale),
				idx_to_complex (x + 5, y, lower_left, scale),
				idx_to_complex (x + 6, y, lower_left, scale),
				idx_to_complex (x + 7, y, lower_left, scale),
			};
			c_real = _mm256_set_ps (
				std::get<7> (c).real, std::get<6> (c).real,
				std::get<5> (c).real, std::get<4> (c).real,
				std::get<3> (c).real, std::get<2> (c).real,
				std::get<1> (c).real, std::get<0> (c).real
			);
			c_imag = _mm256_set_ps (
				std::get<7> (c).imag, std::get<6> (c).imag,
				std::get<5> (c).imag, std::get<4> (c).imag,
				std::get<3> (c).imag, std::get<2> (c).imag,
				std::get<1> (c).imag, std::get<0> (c).imag
			);
		}
		auto result = mandelbrot_avx (c_real, c_imag);
		auto span = image.pixel_span ();

		span[base_idx] = get_color (std::get<0> (result), zooming);
		span[base_idx + 1] = get_color (std::get<1> (result), zooming);
		span[base_idx + 2] = get_color (std::get<2> (result), zooming);
		span[base_idx + 3] = get_color (std::get<3> (result), zooming);
		span[base_idx + 4] = get_color (std::get<4> (result), zooming);
		span[base_idx + 5] = get_color (std::get<5> (result), zooming);
		span[base_idx + 6] = get_color (std::get<6> (result), zooming);
		span[base_idx + 7] = get_color (std::get<7> (result), zooming);
	}

	template<int size = 1>
	inline
	void fill_pixels (size_t x, size_t y, std::vector<pixel_t>& image, const FractalZooming& zooming, const complex_t& lower_left, const std::array<float, 2>& scale) {
		std::array<__m256, size> c_real;
		std::array<__m256, size> c_imag;

		auto c = idx_to_complex_8 (x, y, lower_left, scale);
		c_real[0] = std::get<0> (c);
		c_imag.fill (std::get<1> (c)); // imag will be the same as we fill a row
		for (size_t i = 1; i < size; i++)
		{
			c = idx_to_complex_8 (x + (i * 8), y, lower_left, scale);
			c_real[i] = std::get<0> (c);
		}
		auto result = mandelbrot_avx_multiple<size> (c_real, c_imag);

		auto base_idx = y * _image_width + x;
		auto pixels_count = 8 * size;
		auto overdraw = x + pixels_count < _image_width
			? 0
			: (x + pixels_count) - _image_width;

		std::transform (
			std::begin (result),
			std::end (result) - overdraw,
			std::begin (image) + base_idx,
			[this, &zooming](auto& elem) { return get_color (elem, zooming); }
		);
	}

	inline
	complex_t idx_to_complex (size_t x, size_t y, complex_t lower_left, std::array<float, 2> scale) {
		return lower_left + complex_t{
			x * std::get<0>(scale),
			(_image_height - y - 1) * std::get<1>(scale)
		};
	}

	std::tuple<__m256, __m256> idx_to_complex_8 (size_t x, size_t y, complex_t lower_left, std::array<float, 2> scale) {
		// real = lower_left.real + x * scale.x;
		// imag = lower_left.imag + y * scale.y;
		__m256 scale_real = _mm256_set1_ps (std::get<0>(scale));
		__m256 ll_real = _mm256_set1_ps (lower_left.real());
		__m256 xs = _mm256_add_ps (
			_mm256_set1_ps ((float)x),
			_mm256_set_ps (7, 6, 5, 4, 3, 2, 1, 0)
		);

		__m256 real = _mm256_fmadd_ps (xs, scale_real, ll_real);
		__m256 imag = _mm256_set1_ps (
			(_image_height - y - 1) * std::get<1>(scale) 
			+ lower_left.imag()
		); // y is fix as we calculate a 8 cols in a row

		return std::make_tuple (real, imag);
	}

	inline
	pixel_t get_color (size_t iter_count, const FractalZooming& zooming) {
		return zooming.color_map[iter_count];
	}

	size_t mandelbrot (complex_t c) {
		complex_t z;
		for (size_t i = 0; i < FRACTAL_ITER; i++)
		{
			z = z * z + c;
			auto mag = std::abs(z);
			if (mag > FRACTAL_BOUND) { // divereged
				return i;
			}
		}
		// bounded
		return FRACTAL_ITER - 1;
	}

	std::array<size_t, 8> mandelbrot_avx (__m256 c_real, __m256 c_imag) {
		// 8 32-bit float -> 4 complex numbers
		__m256 const_2 = _mm256_set1_ps (2);
		__m256 z_real = _mm256_set1_ps (0);
		__m256 z_imag = _mm256_set1_ps (0);

		std::array<size_t, 8> result;
		result.fill (FRACTAL_ITER - 1);
		for (size_t i = 0; i < FRACTAL_ITER; i++)
		{
			/*
			z.real = z.real * z.real - z.imag * z.imag + c.real;
			z.imag = 2 * z.real * z.imag + c.imag;
			*/
			__m256 prod = _mm256_mul_ps (z_real, z_imag);
			z_real = _mm256_add_ps (
				_mm256_sub_ps (
					_mm256_mul_ps (z_real, z_real),
					_mm256_mul_ps (z_imag, z_imag)
				),
				c_real
			);

			if constexpr (cpu_ext == FracUseCPUExt::AVX_FMA) {
				z_imag = _mm256_fmadd_ps (prod, const_2, c_imag);
			}
			else {
				z_imag = _mm256_add_ps (
					_mm256_mul_ps (
						prod,
						const_2
					),
					c_imag
				);
			}

			__m256 mag = _mm256_add_ps (
				_mm256_mul_ps (z_real, z_real),
				_mm256_mul_ps (z_imag, z_imag)
			);
			for (size_t j = 0; j < 8; j++)
			{
				if (((float*)&mag)[j] > FRACTAL_BOUND&& result[j] == FRACTAL_ITER - 1) {
					result[j] = i;
				}
			}

			bool all_diverged = std::all_of (
				std::begin (result),
				std::end (result),
				[](const auto& e) {
					return e < FRACTAL_BOUND - 1;
				});
			if (all_diverged)
				return result;
		}
		return result;
	}

	template<int size = 1>
	std::array<size_t, size * 8> mandelbrot_avx_multiple (std::array<__m256, size> c_real, std::array<__m256, size> c_imag) {
		// 8 32-bit float -> 4 complex numbers
		__m256 const_2 = _mm256_set1_ps (2);
		std::array<__m256, size> z_real;
		z_real.fill (_mm256_set1_ps (0));
		std::array<__m256, size> z_imag;
		z_imag.fill (_mm256_set1_ps (0));

		std::array<size_t, size * 8> result;
		result.fill (FRACTAL_ITER - 1);
		for (size_t i = 0; i < FRACTAL_ITER; i++)
		{
			for (size_t j = 0; j < size; j++)
			{
				/*
				z.real = z.real * z.real - z.imag * z.imag + c.real;
				z.imag = 2 * z.real * z.imag + c.imag;
				*/
				__m256 prod = _mm256_mul_ps (z_real[j], z_imag[j]);
				z_real[j] = _mm256_add_ps (
					_mm256_sub_ps (
						_mm256_mul_ps (z_real[j], z_real[j]),
						_mm256_mul_ps (z_imag[j], z_imag[j])
					),
					c_real[j]
				);

				if constexpr (cpu_ext == FracUseCPUExt::AVX_FMA) {
					z_imag[j] = _mm256_fmadd_ps (prod, const_2, c_imag[j]);
				}
				else {
					z_imag[j] = _mm256_add_ps (
						_mm256_mul_ps (
							prod,
							const_2
						),
						c_imag[j]
					);
				}

				__m256 mag = _mm256_add_ps (
					_mm256_mul_ps (z_real[j], z_real[j]),
					_mm256_mul_ps (z_imag[j], z_imag[j])
				);
				for (size_t k = 0; k < 8; k++)
				{
					if (((float*)&mag)[k] > FRACTAL_BOUND&& result[(j * 8) + k] == FRACTAL_ITER - 1) {
						result[(j * 8) + k] = i;
					}
				}

				bool all_diverged = std::all_of (
					std::begin (result),
					std::end (result),
					[](const auto& e) { return e < FRACTAL_BOUND - 1; });
				if (all_diverged)
					return result;
			}
		}
		return result;
	}

	size_t julia (complex_t z) {
		complex_t c{ -0.8f, 0.156f };
		for (size_t i = 0; i < FRACTAL_ITER; i++)
		{
			z = z.square () + c;
			auto mag = z.norm ();
			if (mag > FRACTAL_BOUND) { // divereged - not in set
				return i;
			}
		}
		// bounded - in set
		return FRACTAL_ITER - 1;
	}
};

template<
	FracUseCPUExt cpu_ext = FracUseCPUExt::AVX_FMA,
	int pixels_size = 1,
	FracProgress report_progress = FracProgress::Cout,
	typename parallelizer = typename task_group
>
class FracCPU_GSLP : public FracCPU<cpu_ext, pixels_size> {
	size_t _task_count;

public:
	FracCPU_GSLP (int image_width, int image_height, size_t task_count)
		: FracCPU (image_width, image_height, "FracCPU_GSLP using " + std::string (typeid(parallelizer).name ()) + " (" + std::to_string (task_count) + ")"), _task_count{ task_count } { }

	void execute (const FractalZooming& zooming) {
		AnimatedGif image("zoom.gif", _image_width, _image_height);
		auto delay = 33ms;
		std::vector<pixel_t> frame(_image_width * _image_height);

		_timer.start ("all");

		const size_t partition_size = _image_height / _task_count;
		const size_t partition_remainder = _image_height % _task_count;

		auto lower_left{ zooming.start_lower_left };
		auto upper_right{ zooming.start_upper_right };

		for (size_t i = 0; i < zooming.zoom_steps; i++)
		{
			parallelizer parallelizer;
			auto scale = compute_scale (lower_left, upper_right, _image_width, _image_height);

			for (size_t p = 0; p < _task_count; p++)
			{
				size_t start = p * partition_size;
				size_t end = (p + 1) * partition_size;
				if (p == _task_count - 1)
					end += partition_remainder;

				parallelizer.add ([this, &frame, &lower_left, &scale, &zooming](size_t start, size_t end) {
					for (size_t y = start; y < end; y++)
					{
						fill_row (y, frame, zooming, lower_left, scale);
					}}, start, end);
			}

			zoom_and_re_center_inplace (lower_left, upper_right, zooming);

			parallelizer.join_all ();

			if (zooming.save_images == FractalZooming::SaveImage::ToDisk) {
				// image.to_file (file_name);
				image.append_frame(frame,
					std::chrono::duration_cast<std::chrono::duration<short, std::centi>>(delay), 
					true
				);
			}

			if (report_progress == FracProgress::Cout && i % 10 == 0) {
				std::cout << i << " ";
			}
		}
		if (report_progress == FracProgress::Cout) std::cout << std::endl;

		_timer.stop ();
	}
};

template<
	FracUseCPUExt cpu_ext = FracUseCPUExt::AVX_FMA,
	int pixels_size = 1,
	FracProgress report_progress = FracProgress::Cout,
	typename parallelizer = typename thread_group
>
class FracCPU_GPLS : public FracCPU<cpu_ext, pixels_size> {
	size_t _task_count;

public:
	FracCPU_GPLS (int image_width, int image_height, size_t task_count)
		: FracCPU (image_width, image_height, "FracCPU_GPLS using " + std::string (typeid(parallelizer).name ()) + "(" + std::to_string (task_count) + ")"), _task_count{ task_count } { }

	void execute (const FractalZooming& zooming) {
		std::vector<std::vector<pixel_t>> images;
		for (size_t i = 0; i < _task_count; i++)
		{
			images.emplace_back (_image_width * _image_height);
		}

		_timer.start ("all");

		size_t i = 0;
		std::map<size_t, std::tuple<complex_t, complex_t>> bounds{ get_bounds (zooming) };
		while (i < zooming.zoom_steps)
		{
			parallelizer parallelizer;

			for (size_t t = 0; t < _task_count; t++)
			{
				if (i > zooming.zoom_steps)
					break;

				parallelizer.add ([this, t, i, &bounds, &images, &zooming]() {
					std::vector<pixel_t>& image{ images[t] };

					auto bound = bounds[i];
					auto lower_left = std::get<0> (bound);
					auto scale = compute_scale (lower_left, std::get<1> (bound), _image_width, _image_height);

					for (size_t y = 0; y < _image_height; y++)
					{
						fill_row (y, image, zooming, lower_left, scale);
					}

					if (zooming.save_images == FractalZooming::SaveImage::ToDisk) {
						std::string file_name;
						file_name += "frac_zoom_";
						file_name += std::to_string (i);
						file_name += ".bmp";

						// image.to_file (file_name);
					}
					});

				i++;
			}

			parallelizer.join_all ();

			if (report_progress == FracProgress::Cout && i % _task_count == 0) {
				std::cout << i << " ";
			}
		}
		if (report_progress == FracProgress::Cout) std::cout << std::endl;

		_timer.stop ();
	}
};

template<
	FracUseCPUExt cpu_ext = FracUseCPUExt::AVX_FMA,
	int pixels_size = 1,
	FracProgress report_progress = FracProgress::Cout,
	typename parallelizer = typename task_group
>
class FracCPU_GPLP : public FracCPU<cpu_ext, pixels_size> {
	size_t _image_count;
	size_t _task_count;

public:
	FracCPU_GPLP (int image_width, int image_height, size_t image_count, size_t task_count)
		: FracCPU (image_width, image_height, "FracCPU_GPLP using " + std::string(typeid(parallelizer).name ()) + " (" + std::to_string (image_count) + "/" + std::to_string (task_count) + ")"), _image_count{ image_count }, _task_count{ task_count } { }

	void execute (const FractalZooming& zooming) {
		const size_t partition_size = _image_height / _task_count;
		const size_t partition_remainder = _image_height % _task_count;

		std::vector<std::vector<pixel_t>> images;
		for (size_t i = 0; i < _image_count; i++)
		{
			images.emplace_back (_image_width * _image_height);
		}

		_timer.start ("all");

		size_t i = 0;
		std::map<size_t, std::tuple<complex_t, complex_t>> bounds{ get_bounds (zooming) };
		while (i < zooming.zoom_steps)
		{
			parallelizer parallelizer;

			auto start_i = i;
			for (size_t j = 0; j < _image_count; j++)
			{
				if (i > zooming.zoom_steps)
					break;

				std::vector<pixel_t>& image{ images[j] };
				auto bound = bounds[i];
				auto lower_left = std::get<0> (bound);
				auto scale = compute_scale (lower_left, std::get<1> (bound), _image_width, _image_height);
				for (size_t k = 0; k < _task_count; k++)
				{
					size_t start = k * partition_size;
					size_t end = (k + 1) * partition_size;
					if (k == _task_count - 1)
						end += partition_remainder;

					parallelizer.add ([this, &image, lower_left, scale, &zooming](size_t start, size_t end) {
						for (size_t y = start; y < end; y++)
						{
							fill_row (y, image, zooming, lower_left, scale);
						}}, start, end);
				}

				i++;
			}

			parallelizer.join_all ();

			if (zooming.save_images == FractalZooming::SaveImage::ToDisk) {
				for (size_t j = 0; j < _image_count && start_i + j < zooming.zoom_steps; j++)
				{
					std::string file_name;
					file_name += "frac_zoom_";
					file_name += std::to_string (start_i + j);
					file_name += ".bmp";

					// images[j].to_file (file_name);
				}
			}

			if (report_progress == FracProgress::Cout && i % _image_count == 0) {
				std::cout << i << " ";
			}
		}
		if (report_progress == FracProgress::Cout) std::cout << std::endl;

		_timer.stop ();
	}
};
