add_triton_library(PIMIR
  PIMDialect.cpp

  DEPENDS
  PIMIRIncGen

  LINK_LIBS PUBLIC
  TritonIR
  MLIRIR
  MLIRMemRefDialect
)
