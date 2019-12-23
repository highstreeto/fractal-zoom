#pragma once

#include <complex>

#include "frac_constants.h"
#include "types.h"

using complex_t = std::complex<float>;

struct FractalZooming
{
	enum class SaveImage
	{
		ToDisk,
		No
	};

	complex_t start_lower_left;
	complex_t start_upper_right;
	float zoom;
	size_t zoom_steps;
	complex_t zoom_center;
	SaveImage save_images;
	pixel_t color_map[COLOR_COUNT];
};

inline constexpr pixel_t interpolate(const pixel_t &start, const pixel_t &end, double t)
{
	return pixel_t{
		(unsigned char)(start.r + t * (end.r - start.r)),
		(unsigned char)(start.g + t * (end.g - start.g)),
		(unsigned char)(start.b + t * (end.b - start.b)),
		(unsigned char)(start.a + t * (end.a - start.a)),
	};
}

inline
std::array<float, 2> compute_scale(const complex_t& lower_left, const complex_t& upper_right, const short width, const short height) {
	return std::array<float, 2>{
		(upper_right.real() - lower_left.real()) / width,
		(upper_right.imag() - lower_left.imag()) / height
	};
}

std::tuple<complex_t, complex_t> zoom_and_re_center(const complex_t &lower_left, const complex_t &upper_right, const FractalZooming &zooming)
{
	// Zoom and ...
	auto new_lower_left = lower_left * zooming.zoom;
	auto new_upper_right = upper_right * zooming.zoom;

	auto current_center = (new_lower_left + new_upper_right);
	current_center = complex_t(
		current_center.real() / 2,
		current_center.imag() / 2
	);

	// move towards zoom_center
	auto translate = zooming.zoom_center - current_center;
	new_lower_left = new_lower_left + translate;
	new_upper_right = new_upper_right + translate;

	return std::make_tuple(new_lower_left, new_upper_right);
}

void zoom_and_re_center_inplace(complex_t &lower_left, complex_t &upper_right, const FractalZooming &zooming)
{
	// Zoom and ...
	lower_left = lower_left * zooming.zoom;
	upper_right = upper_right * zooming.zoom;

	auto current_center = (lower_left + upper_right);
	current_center = complex_t(
		current_center.real() / 2,
		current_center.imag() / 2
	);

	// move towards zoom_center
	auto translate = zooming.zoom_center - current_center;
	lower_left = lower_left + translate;
	upper_right = upper_right + translate;
}
