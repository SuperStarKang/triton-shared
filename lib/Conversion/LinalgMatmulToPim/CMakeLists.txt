add_triton_library(LinalgMatmulToPim
  LinalgMatmulToPimPasses.cpp

  DEPENDS
  LinalgMatmulToPimPassIncGen
  PIMIRIncGen

  LINK_LIBS PUBLIC
  PIMIR
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
