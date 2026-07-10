using TensorBoardLogger, Logging, Plots
using Test


@testset "Optional: Plots.jl" begin
logger = TBLogger(test_log_dir*"Plots-jl", tb_overwrite)
p = plot(rand(5))
p2 = plot(rand(10))
with_logger(logger) do
    @info "dispatch" p
    @info "dispatch" twoplots=[p, p2]
end
log_image(logger, "explicit/p", p)

# test serialization (PNG bytes are not deterministic across renders)
data = TensorBoardLogger.preprocess("key", p, Vector())
@test length(data) == 1
@test first(data[1]) == "key"
png = last(data[1])
@test png isa TensorBoardLogger.PngImage
@test png.data[1:8] == UInt8[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]
@test png.attr[:width] > 0
@test png.attr[:height] > 0

# test unpacking of array of plots
data = TensorBoardLogger.preprocess("key", [p, p], Vector())
@test length(data) == 2
@test first(data[1]) == "key/1"
@test first(data[2]) == "key/2"
@test last(data[1]) isa TensorBoardLogger.PngImage
@test last(data[2]) isa TensorBoardLogger.PngImage

close.(values(logger.all_files))
end
