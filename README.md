# fractal-zoom

This was a assignment I did for a course about Multicore programming. It generates a zomming ride toward a given point in the [Mandelbrot](https://en.wikipedia.org/wiki/Mandelbrot_set) set.

## Example

![Fractal Zooming](doc/example_zoom.gif)

## TODOs

* [ ] Implement a more colorful version (using HSV colors and converting to RGB)
  * Vary hue and set saturation and value to constants
* [ ] Put classes into proper namespaces
* [ ] Add GPU impl. (currently this depends on CUDA and is VS CUDA project)