#pragma once

#include <string>
#include <vector>
#include <chrono>

#include "jo_gif.h"
#include "types.h"

class AnimatedGif {
	std::string _file_name;
	size_t _frame_size;
	jo_gif_t _gif;

public:
	AnimatedGif(const std::string& file_name, short width, short height)
		: _file_name{file_name} {
		_gif = jo_gif_start(_file_name.c_str(), width, height, 1, 32);
		_frame_size = width * height;
	}

    void append_frame(std::vector<pixel_t> frame, std::chrono::duration<short, std::centi> delay, bool localPalette = false) {
		if (frame.size() != _frame_size) {
			throw "H"; // TODO Proper exception
		}

		jo_gif_frame(&_gif, reinterpret_cast<unsigned char *>(frame.data()), delay.count(), localPalette);
	}

	~AnimatedGif() {
		jo_gif_end(&_gif);
	}
};