"""
A trait for the ways the mutual information calculation can be parallelized.
"""
abstract type MutualInformationParallelization end

"""
No threading nor vectorization.
"""
struct NoParallelization <: MutualInformationParallelization end

"""
SIMD vectorization.
"""
struct SIMD <: MutualInformationParallelization end
