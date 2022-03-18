using MutualInformationImageRegistration
using Documenter

DocMeta.setdocmeta!(
    MutualInformationImageRegistration,
    :DocTestSetup,
    :(using MutualInformationImageRegistration);
    recursive = true,
)

makedocs(;
    modules = [MutualInformationImageRegistration],
    authors = "Octogonapus <firey45@gmail.com> and contributors",
    repo = "https://github.com/Octogonapus/MutualInformationImageRegistration.jl/blob/{commit}{path}#{line}",
    sitename = "MutualInformationImageRegistration.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://Octogonapus.github.io/MutualInformationImageRegistration.jl",
        assets = String[],
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(; repo = "github.com/Octogonapus/MutualInformationImageRegistration.jl", devbranch = "main")
