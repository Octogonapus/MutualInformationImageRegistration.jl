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

The x and y axes are considered to be the columns and rows of the `full_image`, respectively.
For example, to access this bbox within the full image, this function will do `view(full_image, y:y+h, x:x+w)`.
Furthermore, this function will also do `height, width = size(full_image)`.

```
+---------------------+
|             ^       |
|           y |       |
|             |       |
|             v       |
|         +------+    |
|<------->|   w  |    |
|   x     |      |    |
|         |h     |    |
|         |      |    |
|         +------+    |
|                     |
+---------------------+
```

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

The returned value may be `nothing` if the `moving_bbox` moved outside the bounds of the `full_image`.
Otherwise, the returned value is `shift, best_mi, prev_mis`. `shift` is the computed translation that should be
applied to the `moving_bbox` to align it with the correct section of the `full_image` in the order `[x, y]`.
`best_mi` is the mutual information value associated with that shift.
`prev_mis` is either `missing` or is a cache of the most recently computed mutual information values at each possible
shift, stored as an `OffsetArray`.
"""
function register!(
    mi::MutualInformationContainer,
    full_image::AbstractArray{T,2},
    fixed::AbstractArray{T,2},
    moving_bbox::AbstractVector{Int},
    max_shift_x::Int,
    max_shift_y::Int,
    buffer::AbstractArray{T,2};
    inds = (x1=1,y1=2,x2=3,y2=4),
    set_buffer! = create_set_buffer(inds, max_shift_x, max_shift_y),
    get_buffer_crop = create_get_buffer_crop(inds, max_shift_x, max_shift_y),
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

    # Registration prefiltering runs over all possible shifts, up to the max shifts. Therefore, if the moving image
    # could move out of frame when shifted by the max shifts, we need to abandon it.
    h, w = size(full_image)
    if moving_bbox[inds.x1] - max_shift_x < 1 ||
            moving_bbox[inds.x2] + max_shift_x > w ||
            moving_bbox[inds.y1] - max_shift_y < 1 ||
            moving_bbox[inds.y2] + max_shift_y > h
        return nothing
    end

    while allowjumps[inds.x1] >= -max_shift_x &&
              allowjumps[inds.x2] <= max_shift_x &&
              allowjumps[inds.y1] >= -max_shift_y &&
              allowjumps[inds.y2] <= max_shift_y
        shift, best_mi, prev_mis = register!(
            mi,
            full_image,
            fixed,
            moving_bbox,
            allowjumps[inds.x1]:allowjumps[inds.x2],
            allowjumps[inds.y1]:allowjumps[inds.y2],
            buffer,
            prev_mis;
            set_buffer!,
            get_buffer_crop,
            prefilter_frame_crop!,
        )

        tooclose = 0

        # Expand the maximum shift by expand_increment on any side that is within the expand_border
        if shift[inds.x1] <= allowjumps[inds.x1] + expand_border
            allowjumps[inds.x1] -= expand_increment
            tooclose += 1
        end
        if shift[inds.x1] >= allowjumps[inds.x2] - expand_border
            allowjumps[inds.x2] += expand_increment
            tooclose += 1
        end
        if shift[inds.y1] <= allowjumps[inds.y1] + expand_border
            allowjumps[inds.y1] -= expand_increment
            tooclose += 1
        end
        if shift[inds.y1] >= allowjumps[inds.y2] - expand_border
            allowjumps[inds.y2] += expand_increment
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

The returned value may be `nothing` if the `moving_bbox` moved outside the bounds of the `full_image`.
Otherwise, the returned value is `shift, best_mi, mis`. `shift` is the computed translation that should be
applied to the `moving_bbox` to align it with the correct section of the `full_image` in the order `[x, y]`.
`best_mi` is the mutual information value associated with that shift.
`mis` is a cache of the most recently computed mutual information values at each possible shift, stored as
an `OffsetArray`.
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
    inds = (x1=1,y1=2,x2=3,y2=4),
    set_buffer! = create_set_buffer(inds, maximum(range_x), maximum(range_y)),
    get_buffer_crop = create_get_buffer_crop(inds, maximum(range_x), maximum(range_y)),
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

function create_set_buffer(inds, max_shift_x, max_shift_y)
    return function set_buffer!(buffer, current_frame, moving_bbox)
        x_inds = (moving_bbox[inds.x1]-max_shift_x):(moving_bbox[inds.x2]+max_shift_x)
        y_inds = (moving_bbox[inds.y1]-max_shift_y):(moving_bbox[inds.y2]+max_shift_y)
        buffer .= view(current_frame, y_inds, x_inds)
        nothing
    end
end

function create_get_buffer_crop(inds, max_shift_x, max_shift_y)
    return function get_buffer_crop(buffer, moving_bbox, shift_x, shift_y)
        # Extract the bbox + (shift_x, shift_y). We are extracting it from an array of
        # bbox ± max_shift so we need to transform the coordinates into the right frame by
        # subtracting the origin of the bbox ± max_shift frame.
        x_inds = ((moving_bbox[inds.x1]:moving_bbox[inds.x2]) .+ shift_x) .- (moving_bbox[inds.x1] - max_shift_x) .+ 1
        y_inds = ((moving_bbox[inds.y1]:moving_bbox[inds.y2]) .+ shift_y) .- (moving_bbox[inds.y1] - max_shift_y) .+ 1
        view(buffer, y_inds, x_inds)
    end
end
