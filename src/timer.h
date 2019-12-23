#pragma once

#include <string>
#include <stack>
#include <vector>
#include <tuple>
#include <chrono>

class Timer {
	using Clock = std::chrono::high_resolution_clock;
	using TimePointType = Clock::time_point;
	using DurationType = std::chrono::duration<double>;

	std::stack<std::tuple<std::string, TimePointType>> in_progress;
	std::vector<std::tuple<std::string, DurationType>> finished;

public:
	void start (std::string start_name) {
		in_progress.emplace (start_name, Clock::now ());
	}

	void checkpoint (std::string checkpoint_name) {
		finish_latest ();
		in_progress.emplace (checkpoint_name, Clock::now ());
	}

	void stop () {
		finish_latest ();
	}

	const std::vector<std::tuple<std::string, DurationType>>& times() const {
		return finished;
	}

	DurationType total () const {
		DurationType sum{ 0 };
		for (const auto& e : finished) {
			sum += std::get<1> (e);
		}
		return sum;
	}

	double total_in_ms () const {
		return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(total ()).count();
	}

private:
	void finish_latest () {
		auto latest = in_progress.top ();
		in_progress.pop ();

		finished.emplace_back (
			std::get<0> (latest),
			Clock::now () - std::get<1> (latest)
		);
	}
};