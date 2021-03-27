using BenchmarkTools
using RegisterMI
using RegisterMI: FastHistograms.SingleThreadFixedWidth2DHistogram

function bench_mi()
    fixed = rand(UInt8, 70, 100)
    moving = rand(UInt8, 70, 100)

    mi = MutualInformationContainer(SingleThreadFixedWidth2DHistogram())
    bm = @benchmark mutual_information!($mi, $fixed, $moving)
    display(bm)
    println()

    # Current benchmark:
    # Minimum time should be ~32 μs
    #
    # BenchmarkTools.Trial:
    #   memory estimate:  8.75 KiB
    #   allocs estimate:  5
    #   --------------
    #   minimum time:     32.375 μs (0.00% GC)
    #   median time:      34.322 μs (0.00% GC)
    #   mean time:        37.622 μs (0.36% GC)
    #   maximum time:     1.386 ms (96.61% GC)
    #   --------------
    #   samples:          10000
    #   evals/sample:     1
end

function bench_register()
    MAX_SHIFT = 11
    padding = [-10, -10, 10, 10]
    mi = MutualInformationContainer(SingleThreadFixedWidth2DHistogram())
    full_image = rand(UInt8, 500, 300)
    # Create an asymmetric pattern of black pixels to register against
    view(full_image, 300:320, 200:210) .= 0xff

    fixed = full_image[(300-10):(330+10), (200-10):(220+10)]

    buffer = Array{UInt8}(undef, (size(fixed) .+ (MAX_SHIFT * 2))...)

    expected_x = 5
    expected_y = -5
    moving =
        [300, 200, 330, 220] .+ padding .+ [expected_x, expected_y, expected_x, expected_y]

    bm = @benchmark register!(
        $mi,
        $full_image,
        $fixed,
        $moving,
        $MAX_SHIFT,
        $MAX_SHIFT,
        $buffer,
    )
    display(bm)
    println()

    # Current benchmark:
    # Minimum time should be ~900 μs
    #
    # BenchmarkTools.Trial:
    #   memory estimate:  558.05 KiB
    #   allocs estimate:  332
    #   --------------
    #   minimum time:     1.122 ms (0.00% GC)
    #   median time:      1.229 ms (0.00% GC)
    #   mean time:        1.289 ms (1.36% GC)
    #   maximum time:     4.280 ms (59.71% GC)
    #   --------------
    #   samples:          3877
    #   evals/sample:     1
end
