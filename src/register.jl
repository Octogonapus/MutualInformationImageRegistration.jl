"""
    register!(
        mi::MutualInformationContainer,
        full_image::AbstractArray{T,2},
        fixed::AbstractArray{T,2},
        moving_bbox::AbstractVector{Int},
        max_shift_x::Int,
        max_shift_y::Int,
        buffer::AbstractArray{T,2};
        set_buffer! = (buffer, current_frame, moving_bbox) -> set_buffer!(buffer, current_frame, moving_bbox, max_shift_x, max_shift_y),
        get_buffer_crop = (buffer, moving_bbox, shift_x, shift_y) -> get_buffer_crop(buffer, moving_bbox, shift_x, shift_y, max_shift_x, max_shift_y),
        prefilter_frame_crop! = x -> nothing,
        start_shift_x = 3,
        start_shift_y = 3,
        expand_border = 1,
        expand_increment = 1,
    ) where {T<:Integer}

Calculates the shift that best aligns the `moving_bbox` to the `fixed` image within the `full_image`. At a high level,
this attempts to best match the view of `moving_bbox` inside `full_image` to the `fixed` image. This only considers
shifts along the x and y axes; no rotation is considered. This function incrementally expands the maximum shift horizon
to evaluate as few shifts as possible. At a minimum, all shifts within `± start_shift_x` and `± start_shift_y` are
considered. If the optimal shift falls within `expand_border` of the horizon, the horizon will be expanded by
`expand_increment` and all new shifts will be evaluated. This process repeats until the optimal shift does not fall
within `expand_border` of the horizon, or until the horizon has reached `max_shift_x` and `max_shift_y`.

Adding some padding to `moving_bbox` is a good idea to improve registration stability.
E.g. `my_bbox .+ [-10, -10, 10, 10]`. You will need to determine the best padding value for your data.

The parameter `buffer` is required for temporary storage. Generally, you can define `buffer` as
`Array{T}(undef, (size(fixed) .+ (MAX_SHIFT * 2))...)`. The buffer must have a size which is at least the size of the
`fixed` image expanded by the maximum and minimum x and y shifts on each respective axis. The parameters `set_buffer!`
and `get_buffer_crop` are used to write to and read from this buffer, respectively. There is typically no need to change
these from their defaults.

The parameter `prefilter_frame_crop!` can be specified if you want to apply image filtering before computing the
mutual information between the two images. This function must mutate the image it is given with whatever filtering
operation you implement. Also, the `fixed` image must have the filtering pre-applied. See this package's tests for an
example.
"""
function register!(
    mi::MutualInformationContainer,
    full_image::AbstractArray{T,2},
    fixed::AbstractArray{T,2},
    moving_bbox::AbstractVector{Int},
    max_shift_x::Int,
    max_shift_y::Int,
    buffer::AbstractArray{T,2};
    set_buffer! = (buffer, current_frame, moving_bbox) ->
        set_buffer!(buffer, current_frame, moving_bbox, max_shift_x, max_shift_y),
    get_buffer_crop = (buffer, moving_bbox, shift_x, shift_y) ->
        get_buffer_crop(buffer, moving_bbox, shift_x, shift_y, max_shift_x, max_shift_y),
    prefilter_frame_crop! = x -> nothing,
    start_shift_x = 3,
    start_shift_y = 3,
    expand_border = 1,
    expand_increment = 1,
) where {T<:Integer}
    allowjumps = [-start_shift_x, -start_shift_y, start_shift_x, start_shift_y]
    shift = (0, 0)
    best_mi = 0
    prev_mis::Union{Missing,AbstractArray{Float32,2}} = missing

    while allowjumps[1] >= -max_shift_x &&
              allowjumps[3] <= max_shift_x &&
              allowjumps[2] >= -max_shift_y &&
              allowjumps[4] <= max_shift_y
        shift, best_mi, prev_mis = register!(
            mi,
            full_image,
            fixed,
            moving_bbox,
            allowjumps[1]:allowjumps[3],
            allowjumps[2]:allowjumps[4],
            buffer,
            prev_mis;
            set_buffer!,
            get_buffer_crop,
            prefilter_frame_crop!,
        )

        tooclose = 0

        # Expand the maximum shift by expand_increment on any side that is within the expand_border
        if shift[1] <= allowjumps[1] + expand_border
            allowjumps[1] -= expand_increment
            tooclose += 1
        end
        if shift[1] >= allowjumps[3] - expand_border
            allowjumps[3] += expand_increment
            tooclose += 1
        end
        if shift[2] <= allowjumps[2] + expand_border
            allowjumps[2] -= expand_increment
            tooclose += 1
        end
        if shift[2] >= allowjumps[4] - expand_border
            allowjumps[4] += expand_increment
            tooclose += 1
        end

        # If the shift was not within the expand_border, the shift is a global maximum, so we are done
        if tooclose == 0
            break
        end
    end

    return shift, best_mi, prev_mis
end

"""
    register!(
        mi::MutualInformationContainer,
        full_image::AbstractArray{T,2},
        fixed::AbstractArray{T,2},
        moving_bbox::AbstractVector{Int},
        range_x::AbstractVector{Int},
        range_y::AbstractVector{Int},
        buffer::AbstractArray{T,2},
        prev_mis::Union{Missing,AbstractArray{Float32,2}};
        set_buffer! = (buffer, current_frame, moving_bbox) -> set_buffer!(buffer, current_frame, moving_bbox, maximum(range_x), maximum(range_y)),
        get_buffer_crop = (buffer, moving_bbox, shift_x, shift_y) -> get_buffer_crop(buffer, moving_bbox, shift_x, shift_y, maximum(range_x), maximum(range_y)),
        prefilter_frame_crop! = x -> nothing,
    ) where {T<:Integer}

Calculates the shift that best aligns the `moving_bbox` to the `fixed` image within the `full_image`. At a high level,
this attempts to best match the view of `moving_bbox` inside `full_image` to the `fixed` image. This only considers
shifts along the x and y axes; no rotation is considered. All combinations of shifts within `range_x` and `range_y` are
considered.

Adding some padding to `moving_bbox` is a good idea to improve registration stability.
E.g. `my_bbox .+ [-10, -10, 10, 10]`. You will need to determine the best padding value for your data.

The parameter `buffer` is required for temporary storage between calls to this function because it is assumed that
you will call this function in a loop to register many similar images. Generally, you can define `buffer` as
`Array{T}(undef, (size(fixed) .+ (MAX_SHIFT * 2))...)`. The buffer must have a size which is at least the size of the
`fixed` image expanded by the maximum and minimum x and y shifts on each respective axis. The parameters `set_buffer!`
and `get_buffer_crop` are used to write to and read from this buffer, respectively. There is typically no need to change
these from their defaults.

The parameter `prev_mis` is used to memoize the mutual information calculations if this function is being called
"incrementally" with an expanding shift horizon. If you are calling this function directly (and not in a loop to
gradually expand a maximum shift horizon), you should set this to `missing`, which would cause all combinations of
shifts within `range_x` and `range_y` to be considered.

The parameter `prefilter_frame_crop!` can be specified if you want to apply image filtering before computing the
mutual information between the two images. This function must mutate the image it is given with whatever filtering
operation you implement. Also, the `fixed` image must have the filtering pre-applied. See this package's tests for an
example.
"""
function register!(
    mi::MutualInformationContainer,
    full_image::AbstractArray{T,2},
    fixed::AbstractArray{T,2},
    moving_bbox::AbstractVector{Int},
    range_x::AbstractVector{Int},
    range_y::AbstractVector{Int},
    buffer::AbstractArray{T,2},
    prev_mis::Union{Missing,AbstractArray{Float32,2}};
    set_buffer! = (buffer, current_frame, moving_bbox) ->
        set_buffer!(buffer, current_frame, moving_bbox, maximum(range_x), maximum(range_y)),
    get_buffer_crop = (buffer, moving_bbox, shift_x, shift_y) -> get_buffer_crop(
        buffer,
        moving_bbox,
        shift_x,
        shift_y,
        maximum(range_x),
        maximum(range_y),
    ),
    prefilter_frame_crop! = x -> nothing,
) where {T<:Integer}
    mis = mutual_information!(
        mi,
        fixed,
        buffer,
        full_image,
        moving_bbox,
        range_x,
        range_y,
        prev_mis;
        set_buffer!,
        get_buffer_crop,
        prefilter_frame_crop!,
    )
    best_mi, idx = findmax(mis)
    return idx.I, best_mi, mis
end

function set_buffer!(buffer, current_frame, moving_bbox, max_shift_x, max_shift_y)
    x_inds = (moving_bbox[1]-max_shift_x):(moving_bbox[3]+max_shift_x)
    y_inds = (moving_bbox[2]-max_shift_y):(moving_bbox[4]+max_shift_y)
    @info "set_buffer!" size(buffer) size(view(current_frame, x_inds, y_inds))
    buffer .= view(current_frame, x_inds, y_inds)
    nothing
end

function get_buffer_crop(buffer, moving_bbox, shift_x, shift_y, max_shift_x, max_shift_y)
    # Extract the bbox + (shift_x, shift_y). We are extracting it from an array of
    # bbox ± max_shift so we need to transform the coordinates into the right frame by
    # subtracting the origin of the bbox ± max_shift frame.
    x_inds =
        ((moving_bbox[1]:moving_bbox[3]) .+ shift_x) .- (moving_bbox[1] - max_shift_x) .+ 1
    y_inds =
        ((moving_bbox[2]:moving_bbox[4]) .+ shift_y) .- (moving_bbox[2] - max_shift_y) .+ 1
    view(buffer, x_inds, y_inds)
end
