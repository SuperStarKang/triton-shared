add_triton_library(PimToRuntimeCalls
  PimToRuntimeCallsPass.cpp
  DEPENDS
    PimToRuntimeCallsPassIncGen
  LINK_LIBS PUBLIC
    PIMIR
    MLIRFuncDialect
    MLIRMemRefDialect
    MLIRArithDialect
    MLIRIR
    MLIRPass
)
