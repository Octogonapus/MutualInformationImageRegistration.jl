# RegisterMI

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Octogonapus.github.io/RegisterMI.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Octogonapus.github.io/RegisterMI.jl/dev)
[![Build Status](https://github.com/Octogonapus/RegisterMI.jl/workflows/CI/badge.svg)](https://github.com/Octogonapus/RegisterMI.jl/actions)
[![Coverage](https://codecov.io/gh/Octogonapus/RegisterMI.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Octogonapus/RegisterMI.jl)

RegisterMI performs image registration (i.e. the problem of aligning two similar images) using
[mutual information](https://en.wikipedia.org/wiki/Mutual_information).
This package is meant to be used to quickly register many relatively small images within a larger image.
This package only supports computing a translation to register images; rotation and other warping is a non-goal.

## Example

```julia
using RegisterMI, FastHistogram, Random

# Create the container used to hold intermediate variables for registration
mi = MutualInformationContainer(SingleThreadFixedWidth2DHistogram())

# Create the full image that the smaller images to register will be pulled from
full_image = rand(UInt8, 500, 300)

# The fixed image is the image that the other images are registered against
fixed = full_image[(300-10):(330+10), (200-10):(220+10)]

# A buffer is needed to hold intermediate data
buffer = Array{UInt8}(undef, (size(fixed) .+ (MAX_SHIFT * 2))...)

# The max shift is the maximum search range
MAX_SHIFT = 11
# Padding is used to grow the bbox for higher quality registration
padding = [-10, -10, 10, 10]

# Introduce a random shift to the moving bbox
expected_x = rand(-5:5)
expected_y = rand(-5:5)

# Register the image given by the bbox (called the moving bbox) against the fixed image
shift, mm, mms = register!(
    mi,
    full_image,
    fixed,
    [300, 200, 330, 220] .+ padding .+ [expected_x, expected_y, expected_x, expected_y],
    MAX_SHIFT,
    MAX_SHIFT,
    buffer
)

# The shift we get out should be equal and opposite of the shift we applied
@test shift == (-expected_x, -expected_y)
```

## References

The mutual information implementation is based on [this work](https://matthew-brett.github.io/teaching/mutual_information.html) by Matthew Brett.

## Benchmarks

With two 50x40 8-bit images and an expected shift of (-5, 5), RegisterMI runs in 1.2 ms.
