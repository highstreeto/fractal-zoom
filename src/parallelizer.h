#include <vector>
#include <thread>
#include <future>

class thread_group {
    std::vector <std::thread> _in_progress;

public:
    explicit thread_group () = default;

    // delete copy ctor and assign. as this in progress threads can not be copied
    thread_group (const thread_group&) = delete;
    thread_group& operator= (const thread_group&) = delete;

    thread_group (thread_group &&) = default;
    thread_group& operator= (thread_group &&) = default;

    template <typename TFunc, typename... TArgs>
    void add (TFunc&& func, TArgs&&... args) {
        _in_progress.emplace_back (
            std::forward <TFunc> (func),
            std::forward <TArgs> (args)...
        );
    }

    void join_all () {
        for (auto & t : _in_progress)
            if (t.joinable ()) t.join ();
    }

    ~thread_group () {
        join_all ();
    }
};

class task_group {
    std::vector <std::future <void>> _in_progress;

public:
    explicit task_group () = default;

    // delete copy ctor and assign. as this in progress futures can not be copied
    task_group (const task_group&) = delete;
    task_group& operator= (const task_group&) = delete;

    task_group (task_group &&) = default;
    task_group& operator= (task_group &&) = default;

    template <typename TFunc, typename... TArgs>
    void add (TFunc&& func, TArgs&&... args) {
        _in_progress.push_back (
            std::async (std::launch::async, std::forward <TFunc> (func),
            std::forward <TArgs> (args)...)
        );
    }

    void join_all () {
        for (auto & f : _in_progress) f.wait ();
    }
};