using MutualInformationImageRegistration
using Random, ImageFiltering, ComputationalResources, Test
using MutualInformationImageRegistration.FastHistograms
using JLD2, BenchmarkTools

const MAX_SHIFT = 11
const PADDING = [-10, -10, 10, 10]
const ALL_PARALLELIZATIONS =
    [MutualInformationImageRegistration.NoParallelization(), MutualInformationImageRegistration.SIMD()]

@testset "MutualInformationImageRegistration.jl" begin
    @testset "register without filtering ($p)" for p in ALL_PARALLELIZATIONS
        mi = MutualInformationContainer(
            create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                FastHistograms.NoParallelization(),
                [(0x00, 0xff, 8), (0x00, 0xff, 8)],
            ),
            p,
        )
        full_image = rand(UInt8, 500, 300)
        view(full_image, 300:330, 200:220) .= 0xff

        fixed = full_image[(300-10):(330+10), (200-10):(220+10)]

        buffer = Array{UInt8}(undef, (size(fixed) .+ (MAX_SHIFT * 2))...)

        for i = 1:1000
            expected_x = rand(-5:5)
            expected_y = rand(-5:5)

            shift, mm, mms = register!(
                mi,
                full_image,
                fixed,
                [300, 200, 330, 220] .+ PADDING .+ [expected_x, expected_y, expected_x, expected_y],
                MAX_SHIFT,
                MAX_SHIFT,
                buffer,
            )

            # The shift we get out should be equal and opposite of the shift we applied
            @test shift == (-expected_x, -expected_y)
            if shift != (-expected_x, -expected_y)
                jldsave(
                    "register without filtering fail.jld2";
                    mi,
                    full_image,
                    fixed,
                    buffer,
                    expected_x,
                    expected_y,
                    shift,
                    mm,
                    mms,
                )
                break
            end
        end
    end

    @testset "register with filtering ($p)" for p in ALL_PARALLELIZATIONS
        function prefilter!(img::Array{UInt8,2})
            buf = Float32.(img)
            buf =
                imfilter(CPU1(Algorithm.FIR()), buf, (centered(ones(2, 1) ./ 2), centered(ones(1, 2) ./ 2)), "reflect")
            img .= round.(UInt8, buf)
            return nothing
        end

        mi = MutualInformationContainer(
            create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                FastHistograms.NoParallelization(),
                [(0x00, 0xff, 8), (0x00, 0xff, 8)],
            ),
            p,
        )
        full_image = rand(UInt8, 500, 300)
        view(full_image, 300:330, 200:220) .= 0xff

        fixed = full_image[(300-10):(330+10), (200-10):(220+10)]

        prefilter!(fixed)

        buffer = Array{UInt8}(undef, (size(fixed) .+ MAX_SHIFT * 2)...)

        for i = 1:1000
            expected_x = rand(-5:5)
            expected_y = rand(-5:5)

            shift, mm, mms = register!(
                mi,
                full_image,
                fixed,
                [300, 200, 330, 220] .+ PADDING .+ [expected_x, expected_y, expected_x, expected_y],
                MAX_SHIFT,
                MAX_SHIFT,
                buffer;
                prefilter_frame_crop! = prefilter!,
            )

            # The shift we get out should be equal and opposite of the shift we applied
            @test shift == (-expected_x, -expected_y)
            if shift != (-expected_x, -expected_y)
                jldsave(
                    "register with filtering fail.jld2";
                    mi,
                    full_image,
                    fixed,
                    buffer,
                    expected_x,
                    expected_y,
                    shift,
                    mm,
                    mms,
                )
                break
            end
        end
    end

    @testset "computing mutual_information doesn't allocate ($p)" for p in ALL_PARALLELIZATIONS
        mi = MutualInformationContainer(
            create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                FastHistograms.NoParallelization(),
                [(0x00, 0xff, 8), (0x00, 0xff, 8)],
            ),
            p,
        )

        x = rand(UInt8, 500, 300)
        y = rand(UInt8, 500, 300)

        @test 0 == @ballocated mutual_information!($mi, $x, $y)
    end

    @testset "default parallelization is NoParallelization" begin
        mi = MutualInformationContainer(
            create_fast_histogram(
                FastHistograms.FixedWidth(),
                FastHistograms.Arithmetic(),
                FastHistograms.NoParallelization(),
                [(0x00, 0xff, 8), (0x00, 0xff, 8)],
            ),
        )
        @test MutualInformationImageRegistration.MutualInformationParallelization(mi) ==
              MutualInformationImageRegistration.NoParallelization()
    end
end
