add_triton_library(LinalgMatmulToUpmem
  LinalgMatmulToUpmemPass.cpp

  DEPENDS
  LinalgMatmulToUpmemPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRIR
  MLIRPass
  MLIRSupport
)
