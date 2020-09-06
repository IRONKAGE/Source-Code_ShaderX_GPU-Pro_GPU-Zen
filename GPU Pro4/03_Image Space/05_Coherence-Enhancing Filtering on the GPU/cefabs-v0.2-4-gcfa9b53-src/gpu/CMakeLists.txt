FILE(GLOB sources *.cpp *.h *.cu)
CUDA_ADD_LIBRARY(gpu ${sources})
SOURCE_GROUP(src REGULAR_EXPRESSION "c$|cpp$|hpp$|h$|ui$|qrc$|cu$")
