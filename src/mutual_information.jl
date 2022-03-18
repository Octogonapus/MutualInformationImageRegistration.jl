struct MutualInformationContainer{H}
    hist::H
    pxy::Array{Float32,2}
    px::Array{Float32,2}
    py::Array{Float32,2}
    px_py::Array{Float32,2}
    nzs::BitArray{2}

    function MutualInformationContainer(hist::H) where {H}
        pxy = counts(hist) ./ sum(counts(hist))
        px = sum(pxy, dims = 2)
        py = sum(pxy, dims = 1)
        px_py = px .* py
        nzs = pxy .> 0

        new{H}(hist, pxy, px, py, px_py, nzs)
    end
end

function _mutual_information!(mi::MutualInformationContainer)
    mi.pxy .= counts(mi.hist) ./ sum(counts(mi.hist))
    sum!(mi.px, mi.pxy)
    sum!(mi.py, mi.pxy)
    mi.px_py .= mi.px .* mi.py
    return _compute_mi_sum(mi.pxy, mi.px_py)
end

function _compute_mi_sum(pxy, px_py)
    _sum = 0
    @turbo for i in eachindex(pxy)
        _sum += pxy[i] > 0 ? pxy[i] * log(pxy[i] / px_py[i]) : 0
    end
    return _sum
end

"""
    mutual_information!(mi::MutualInformationContainer, x, y)

Computes the mutual information between the two variables `x` and `y`. The histogram within `mi` must be of the correct
type to handle the formats of `x` and `y`.
"""
function mutual_information!(mi::MutualInformationContainer, x, y)
    zero!(mi.hist)
    increment_bins!(mi.hist, x, y)
    _mutual_information!(mi)
end

"""
    mutual_information!(
        mi::MutualInformationContainer,
        fixed,
        buffer,
        full_image,
        moving_bbox,
        range_x,
        range_y,
        ::Missing;
        set_buffer!,
        get_buffer_crop,
        prefilter_frame_crop! = x -> nothing,
    )

Calculates the mutual information of two images at all shifts within the `range_x` and `range_y`. The `fixed` image
must already be filtered. This will set the `buffer` and filter its contents using `prefilter_frame_crop!`.
"""
function mutual_information!(
    mi::MutualInformationContainer,
    fixed,
    buffer,
    full_image,
    moving_bbox,
    range_x,
    range_y,
    ::Missing;
    set_buffer!,
    get_buffer_crop,
    prefilter_frame_crop! = x -> nothing,
)
    mis = OffsetArray(
        Array{Float32}(undef, length(range_x), length(range_y)),
        range_x,
        range_y,
    )
    fixed_vec = vec(fixed)

    # Crop and prefilter a section of `current_frame` big enough to handle the shift extents.
    set_buffer!(buffer, full_image, moving_bbox)
    prefilter_frame_crop!(buffer)

    @inbounds for shift_x in range_x
        @inbounds for shift_y in range_y
            moving_vec = vec(get_buffer_crop(buffer, moving_bbox, shift_x, shift_y))
            mis[shift_x, shift_y] = mutual_information!(mi, fixed_vec, moving_vec)
        end
    end

    return mis
end

"""
    mutual_information!(
        mi::MutualInformationContainer,
        fixed,
        buffer,
        ::Any,
        moving_bbox,
        range_x,
        range_y,
        prev_mis::AbstractArray{Float32,2};
        get_buffer_crop,
        kwargs...
    )

Calculates the mutual information of two images at all shifts within the `range_x` and `range_y`. Warm-starts the
evaluation using previous results (`prev_mis`; the return value from a previous call of this function) and using the
previously set and filtered contents of the `buffer`.
"""
function mutual_information!(
    mi::MutualInformationContainer,
    fixed,
    buffer,
    ::Any,
    moving_bbox,
    range_x,
    range_y,
    prev_mis::AbstractArray{Float32,2};
    get_buffer_crop,
    kwargs...,
)
    mis = OffsetArray(
        Array{Float32}(undef, length(range_x), length(range_y)),
        range_x,
        range_y,
    )

    prev_range_x = axes(prev_mis, 1)
    prev_range_y = axes(prev_mis, 2)

    fixed_vec = vec(fixed)

    # No need to extract and prefilter a crop to fill the buffer here because it is done in `mutual_information!`
    # where `prev_mis` is of type `Missing`. We just reuse the prefiltered crop here.

    @inbounds for shift_x in range_x
        @inbounds for shift_y in range_y
            if shift_x ∈ prev_range_x && shift_y ∈ prev_range_y
                mis[shift_x, shift_y] = prev_mis[shift_x, shift_y]
            else
                moving_vec = vec(get_buffer_crop(buffer, moving_bbox, shift_x, shift_y))
                mis[shift_x, shift_y] = mutual_information!(mi, fixed_vec, moving_vec)
            end
        end
    end

    return mis
end
