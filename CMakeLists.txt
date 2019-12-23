cmake_minimum_required (VERSION 3.14)

if(NOT DEFINED ENV{VCPKG_ROOT})
  message("INFO: VCPKG_ROOT not defined - auto-detect will NOT work!")
endif()

if(DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "")
endif()

project("FractalZoom" LANGUAGES CXX)
# TODO Handle other compilers (this will only work with MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")

add_executable (FractalZoom
  "lib/jo_gif.cpp"
  "src/main.cpp"
)
target_compile_features(FractalZoom PUBLIC cxx_std_17)
target_include_directories(FractalZoom PRIVATE "./include")
