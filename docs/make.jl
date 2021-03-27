using RegisterMI
using Documenter

DocMeta.setdocmeta!(RegisterMI, :DocTestSetup, :(using RegisterMI); recursive=true)

makedocs(;
    modules=[RegisterMI],
    authors="Octogonapus <firey45@gmail.com> and contributors",
    repo="https://github.com/Octogonapus/RegisterMI.jl/blob/{commit}{path}#{line}",
    sitename="RegisterMI.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://Octogonapus.github.io/RegisterMI.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Octogonapus/RegisterMI.jl",
    devbranch="main",
)
